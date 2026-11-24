"""Worker run-loop shutdown/EOF regression tests.

The first-op batching regression is already fixed on upstream main. These tests
cover the two remaining edges: an EOF while draining must not discard the
already-received batch, and a drained shutdown sentinel must not call subclass
shutdown before ``_worker_process`` performs its single cleanup in ``finally``.
"""

import types

import pytest


pytestmark = pytest.mark.unit
pytest.importorskip("flexkv.c_ext", reason="requires compiled c_ext")


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _FakeOp:
    def __init__(self, op_id):
        from flexkv.common.transfer import TransferType

        self.transfer_op_id = op_id
        self.transfer_graph_id = 0
        self.transfer_type = TransferType.H2D
        self.valid_block_num = 1


class _EOFAfterDrainPipe:
    def __init__(self, items):
        self.items = list(items)

    def poll(self, timeout=0):
        if self.items:
            return True
        if timeout == 0:
            return True
        raise EOFError

    def recv(self):
        if self.items:
            return self.items.pop(0)
        raise EOFError


class _SentinelPipe:
    def __init__(self, items):
        self.items = list(items)

    def poll(self, timeout=0):
        if self.items:
            return True
        if timeout and timeout > 0:
            raise EOFError
        return False

    def recv(self):
        return self.items.pop(0)


def _worker(pipe, monkeypatch):
    # ``TransferWorkerBase`` and its run loop live in workers/runtime.py now;
    # flexkv.transfer.worker is a re-export façade, so it has no ``trace``
    # module attribute to patch. The loop under test is unchanged.
    from flexkv.transfer.workers import runtime as worker_module

    monkeypatch.setattr(
        worker_module.trace,
        "build_worker_metrics",
        lambda *_args, **_kwargs: {"test": True},
    )
    launched = []
    shutdowns = []
    instance = types.SimpleNamespace(
        worker_id=0,
        transfer_conn=pipe,
        finished_ops_queue=_FakeQueue(),
        kv_dim=2,
        _bytes_per_block=1,
        launch_transfer=lambda op: launched.append(op.transfer_op_id) or True,
        shutdown=lambda: shutdowns.append(True),
        has_deferred_completions=lambda: False,
        _trace_bytes_per_block=lambda: 1,
        _drain_failed_transfer=lambda: True,
        _undrained_transfer_failures={},
    )
    instance._failed_transfer_result = types.MethodType(
        worker_module.TransferWorkerBase._failed_transfer_result, instance)
    instance.run = types.MethodType(
        worker_module.TransferWorkerBase.run, instance)
    return instance, launched, shutdowns


def test_eof_during_drain_runs_received_batch(monkeypatch):
    worker, launched, shutdowns = _worker(
        _EOFAfterDrainPipe([_FakeOp(1), _FakeOp(2)]), monkeypatch)

    worker.run()

    assert launched == [1, 2]
    assert [item[0] for item in worker.finished_ops_queue.items] == [1, 2]
    assert shutdowns == []


def test_drained_sentinel_defers_shutdown_to_worker_process(monkeypatch):
    worker, launched, shutdowns = _worker(
        _SentinelPipe([_FakeOp(1), _FakeOp(2), None]), monkeypatch)

    worker.run()

    assert launched == [1, 2]
    assert [item[0] for item in worker.finished_ops_queue.items] == [1, 2]
    assert shutdowns == []


def _gpu_failure_fence(worker, monkeypatch, *, fail_group=None, device_drained=True):
    from unittest.mock import Mock

    from flexkv.transfer.workers import gpu_cpu

    events = []

    def group(name):
        def wait():
            events.append(name)
            if name == fail_group:
                raise RuntimeError("injected stream drain failure")
        return types.SimpleNamespace(wait_all_streams=Mock(side_effect=wait))

    region, first, second = group("region"), group("first"), group("second")
    worker.region_batch = region
    worker.tp_transfer_thread_group = first
    worker._pools = {"full": types.SimpleNamespace(thread_groups=[
        {"tp_thread_group": first}, {"tp_thread_group": second},
    ])}
    worker._device_ids = [3, 5]
    worker._cuda_device_ids = {5}

    def device_sync(devices, timeout, *, name):
        assert devices == [3, 5] and 0 < timeout <= 30.0
        assert name == "flexkv-worker-failure"
        events.append("devices")
        return device_drained

    monkeypatch.setattr(gpu_cpu, "synchronize_cuda_devices", device_sync)
    monkeypatch.setattr(gpu_cpu.torch.cuda, "is_initialized", lambda: True)
    worker._drain_failed_transfer = types.MethodType(
        gpu_cpu.GPUCPUTransferWorker._drain_failed_transfer, worker)
    return events, (region, first, second)


def test_partial_cuda_launch_failure_is_fenced_before_reporting_terminal(monkeypatch):
    first, second = _FakeOp(1), _FakeOp(2)
    worker, launched, _ = _worker(_SentinelPipe([first, second, None]), monkeypatch)
    events, groups = _gpu_failure_fence(worker, monkeypatch)

    def launch(op):
        launched.append(op.transfer_op_id)
        if op is first:
            events.append("partial_launch")
            raise RuntimeError("partial CUDA launch")
        events.append("next_launch")
        return True

    worker.launch_transfer = launch
    original_put = worker.finished_ops_queue.put

    def report(item):
        events.append(("report", item[0], item[1]))
        original_put(item)

    worker.finished_ops_queue.put = report
    worker.run()
    assert events == [
        "partial_launch", "region", "first", "second", "devices",
        ("report", 1, False), "next_launch", ("report", 2, True),
    ]
    assert launched == [1, 2]
    for group in groups:
        group.wait_all_streams.assert_called_once()
    assert not worker._undrained_transfer_failures


@pytest.mark.parametrize("fail_group, device_drained", [
    ("region", True), ("first", True), ("second", True), (None, False),
])
def test_failed_cuda_fence_retains_owners_without_any_terminal_completion(
    monkeypatch, fail_group, device_drained,
):
    first, second = _FakeOp(1), _FakeOp(2)
    worker, launched, _ = _worker(_SentinelPipe([first, second, None]), monkeypatch)
    events, groups = _gpu_failure_fence(
        worker, monkeypatch, fail_group=fail_group, device_drained=device_drained,
    )

    def launch(op):
        launched.append(op.transfer_op_id)
        raise RuntimeError("partial CUDA launch")

    worker.launch_transfer = launch
    worker.run()
    assert launched == [1]
    assert events == ["region", "first", "second", "devices"]
    assert worker.finished_ops_queue.items == []
    assert worker._undrained_transfer_failures[1][0] is first
    assert worker._undrained_transfer_failures[2][0] is second
    for group in groups:
        group.wait_all_streams.assert_called_once()


def test_failure_drain_hook_exception_is_fail_closed(monkeypatch):
    from unittest.mock import Mock

    first, second = _FakeOp(1), _FakeOp(2)
    worker, launched, _ = _worker(_SentinelPipe([first, second, None]), monkeypatch)
    worker.launch_transfer = Mock(side_effect=RuntimeError("partial launch"))
    worker._drain_failed_transfer = Mock(side_effect=RuntimeError("fence owner lost"))
    worker.run()
    assert worker.launch_transfer.call_count == 1
    assert worker.finished_ops_queue.items == []
    assert worker._undrained_transfer_failures[1][0] is first
    assert worker._undrained_transfer_failures[2][0] is second


def test_gpu_failure_fence_includes_all_pool_groups_and_registered_devices(monkeypatch):
    worker, _, _ = _worker(_SentinelPipe([None]), monkeypatch)
    events, groups = _gpu_failure_fence(worker, monkeypatch)
    worker.region_batch = None
    worker._pools["swa"] = types.SimpleNamespace(thread_groups=[
        {"tp_thread_group": groups[2]},
    ])
    assert worker._drain_failed_transfer()
    assert events == ["first", "second", "devices"]
    groups[0].wait_all_streams.assert_not_called()
    groups[1].wait_all_streams.assert_called_once()
    groups[2].wait_all_streams.assert_called_once()


def test_missing_cuda_owner_cannot_be_treated_as_a_successful_failure_fence(monkeypatch):
    from flexkv.transfer.workers import gpu_cpu

    worker, _, _ = _worker(_SentinelPipe([None]), monkeypatch)
    events, _ = _gpu_failure_fence(worker, monkeypatch)
    monkeypatch.setattr(gpu_cpu.torch.cuda, "is_initialized", lambda: False)
    assert not worker._drain_failed_transfer()
    assert events == ["region", "first", "second"]


def test_hung_stream_fence_is_bounded_and_owner_is_recorded_before_wait(monkeypatch):
    import threading
    import time

    from flexkv.transfer.workers import gpu_cpu

    first, second = _FakeOp(1), _FakeOp(2)
    worker, _, _ = _worker(_SentinelPipe([first, second, None]), monkeypatch)
    _, groups = _gpu_failure_fence(worker, monkeypatch)
    release, entered, completed = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(gpu_cpu, "_FAILED_TRANSFER_DRAIN_TIMEOUT_S", 0.05)

    def hang():
        assert worker._undrained_transfer_failures[1][0] is first
        entered.set()
        release.wait()

    groups[0].wait_all_streams.side_effect = hang
    groups[2].wait_all_streams.side_effect = completed.set

    def launch(op):
        raise RuntimeError("partial launch before stream hang")

    worker.launch_transfer = launch
    started = time.monotonic()
    try:
        worker.run()
        assert time.monotonic() - started < 2.0
        assert entered.is_set()
        assert worker.finished_ops_queue.items == []
        assert worker._undrained_transfer_failures[1][0] is first
        assert worker._undrained_transfer_failures[2][0] is second
    finally:
        release.set()
        assert completed.wait(2.0)


def test_failure_fence_thread_start_error_does_not_publish_terminal(monkeypatch):
    from flexkv.transfer.workers import gpu_cpu

    first = _FakeOp(1)
    worker, _, _ = _worker(_SentinelPipe([first, None]), monkeypatch)
    events, _ = _gpu_failure_fence(worker, monkeypatch)

    def fail_start(thread):
        raise RuntimeError("failure fence thread unavailable")

    monkeypatch.setattr(gpu_cpu.threading.Thread, "start", fail_start)
    worker.launch_transfer = lambda op: (_ for _ in ()).throw(RuntimeError("partial launch"))
    worker.run()
    assert worker.finished_ops_queue.items == []
    assert worker._undrained_transfer_failures[1][0] is first
    assert events == []


def test_fence_hook_exception_retains_original_launch_frame_keepalives(monkeypatch):
    import gc
    import weakref

    from unittest.mock import Mock

    class Keepalive:
        pass

    first, second = _FakeOp(1), _FakeOp(2)
    worker, _, _ = _worker(_SentinelPipe([first, second, None]), monkeypatch)
    references = []

    def launch(op):
        launch_input = Keepalive()
        references.append(weakref.ref(launch_input))
        raise RuntimeError("partial launch owns local input")

    worker.launch_transfer = launch
    worker._drain_failed_transfer = Mock(side_effect=RuntimeError("fence hook lost"))
    worker.run()
    gc.collect()
    assert references[0]() is not None
    errors = worker._undrained_transfer_failures[1][1]
    assert str(errors[0]) == "partial launch owns local input"
    assert str(errors[1]) == "fence hook lost"
    assert worker.finished_ops_queue.items == []
