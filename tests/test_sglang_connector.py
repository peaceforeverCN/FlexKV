from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import flexkv.integration.sglang.connector as connector_module
from flexkv.integration.sglang.comm import FlexKVScatterChannel
from flexkv.integration.sglang.connector import FlexKVConnector

pytestmark = pytest.mark.unit


def test_lookup_accepts_sglang_array_token_ids():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector.kv_manager = None
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=False,
        needs_sync=True,
        scatter=MagicMock(return_value={"task_id": -1, "hit": 0}),
    )
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    connector._new_op_context = MagicMock(
        return_value=SimpleNamespace(task_id=-1)
    )
    connector._log_cache_op = MagicMock()
    token_ids = array("q", [1, 2, 3, 4])

    assert connector.lookup_kv(
        token_ids,
        torch.ones(len(token_ids), dtype=torch.bool),
        rid="request",
    ) == (-1, 0)
    connector._log_cache_op.assert_called_once()


def test_registration_distinguishes_physical_and_global_kv_heads():
    for physical_heads, global_heads in ((8, 32), (1, 4), (1, 1)):
        connector = FlexKVConnector.__new__(FlexKVConnector)
        connector.model_config = SimpleNamespace(
            kv_dim=2, num_kv_heads=global_heads
        )
        connector.rank_info = SimpleNamespace(num_layers_per_pp_stage=2)
        connector.page_size = 16
        connector.tp_client = MagicMock()
        connector._label = "test"
        kv_caches = [torch.empty(160, physical_heads, 128)] * 4

        connector._register_standard_to_server(kv_caches)

        layout = connector.tp_client.register_to_server.call_args.kwargs[
            "kv_layout"
        ]
        assert layout.num_head == physical_heads
        assert layout.num_kv_heads == global_heads


def _idle_connector(chunked_prefetch):
    return SimpleNamespace(
        _sync_ctx=SimpleNamespace(is_sync_leader=True),
        kv_manager=MagicMock(),
        _launched_load_tids=[],
        _chunked_prefetch=chunked_prefetch,
    )


def test_idle_connector_polls_detached_tasks():
    connector = _idle_connector(chunked_prefetch=False)
    FlexKVConnector.drain_launched_loads(connector)
    connector.kv_manager.try_wait.assert_called_once_with(task_ids=[])
    connector.kv_manager.wait.assert_not_called()


def test_idle_connector_rate_limits_engine_nudges(monkeypatch):
    connector = _idle_connector(chunked_prefetch=False)
    now = [100.0]
    monkeypatch.setattr(connector_module.time, "monotonic", lambda: now[0])

    for _ in range(1000):
        FlexKVConnector.drain_launched_loads(connector)
    assert connector.kv_manager.try_wait.call_count == 1

    now[0] += connector_module._IDLE_DRIVE_INTERVAL_S * 0.5
    FlexKVConnector.drain_launched_loads(connector)
    assert connector.kv_manager.try_wait.call_count == 1

    now[0] += connector_module._IDLE_DRIVE_INTERVAL_S
    FlexKVConnector.drain_launched_loads(connector)
    assert connector.kv_manager.try_wait.call_count == 2


def test_idle_chunked_connector_leaves_draining_to_task_runtime():
    connector = _idle_connector(chunked_prefetch=True)
    for _ in range(10):
        FlexKVConnector.drain_launched_loads(connector)
    connector.kv_manager.try_wait.assert_not_called()
    connector.kv_manager.wait.assert_not_called()


def test_single_launched_load_is_polled_until_fully_complete():
    manager = MagicMock()
    manager.wait.return_value = {}
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._sync_ctx = SimpleNamespace(is_sync_leader=True)
    connector.kv_manager = manager
    connector._launched_load_tids = [7]

    connector.drain_launched_loads()

    manager.wait.assert_called_once_with([7], timeout=0.0, completely=True)
    assert connector._launched_load_tids == [7]


def test_nonleader_does_not_poll_idle_or_launched_loads():
    for task_ids in ([], [7]):
        manager = MagicMock()
        connector = SimpleNamespace(
            _sync_ctx=SimpleNamespace(is_sync_leader=False),
            kv_manager=manager,
            _launched_load_tids=task_ids,
        )

        FlexKVConnector.drain_launched_loads(connector)

        assert not manager.mock_calls


def test_legacy_cancel_prefetch_cancels_lookup_and_prefetch_tasks():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._chunked_prefetch = False
    connector.kv_manager = MagicMock()
    connector._sync_ctx = SimpleNamespace(is_sync_leader=True)
    connector._pending_lookups = {"r": 11}
    connector._pending_lookup_contexts = {}
    connector._ongoing_prefetches = {"r": 22}
    connector._prefetch_planned_tokens = {"r": 64}
    connector._prefetch_loaded_tokens = {}
    connector._prefetch_contexts = {}
    connector._log_cache_op = MagicMock()

    connector.cancel_prefetch("r")
    connector.cancel_prefetch("r")  # Repeated cancellation must be harmless.

    assert [c.args[0] for c in connector.kv_manager.cancel.call_args_list] == [
        [11],
        [22],
    ]
    assert not connector._pending_lookups and not connector._ongoing_prefetches


def _restore_mapping_connector(swa_pool, translate=None):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._swa_kv_pool = swa_pool
    connector._kvcache = SimpleNamespace(
        **({} if translate is None else {"translate_loc_from_full_to_swa": translate})
    )
    return connector


def test_restore_mappings_without_an_swa_pool_return_only_the_full_mapping():
    connector = _restore_mapping_connector(None)
    slots = torch.arange(4, dtype=torch.int64)

    full, swa = connector._restore_slot_mappings(slots)

    assert swa is None
    assert torch.equal(full, slots)
    assert full.dtype == torch.int64


def test_restore_mappings_keep_only_the_mapped_swa_tail():
    # The allocator maps the SWA sidecar for the tail page alone, so the
    # translation of the earlier slots reads as unmapped and must be dropped
    # before the mapping reaches FlexKV.
    translated = torch.tensor([0, 0, 8, 9], dtype=torch.int64)
    connector = _restore_mapping_connector(
        object(), translate=lambda _slots: translated
    )
    slots = torch.arange(40, 44, dtype=torch.int64)

    full, swa = connector._restore_slot_mappings(slots)

    assert torch.equal(full, slots)
    assert torch.equal(swa, torch.tensor([8, 9], dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_restore_mappings_land_in_pinned_memory_on_one_host_wait():
    # Pageable copies were two separate cudaStreamSynchronize calls on the
    # scheduler thread. The replacement must still produce detached, own-
    # storage tensors: they are pickled onto the FlexKV control socket, where
    # a view would carry its whole base buffer.
    translated = torch.cat(
        [torch.zeros(6, dtype=torch.int64), torch.tensor([3, 4], dtype=torch.int64)]
    ).cuda()
    connector = _restore_mapping_connector(
        object(), translate=lambda _slots: translated
    )
    slots = torch.arange(100, 108, dtype=torch.int64, device="cuda")

    full, swa = connector._restore_slot_mappings(slots)

    assert full.device.type == "cpu" and full.is_pinned()
    assert torch.equal(full, torch.arange(100, 108, dtype=torch.int64))
    assert torch.equal(swa, torch.tensor([3, 4], dtype=torch.int64))
    assert swa.numel() * swa.element_size() == swa.untyped_storage().nbytes()


def _staged_connector(*, leader=True, scatter=None, defer=False):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.defer_retrieve_wait = defer
    connector.enable_layerwise = False
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=leader, needs_sync=scatter is not None, scatter=scatter,
        all_reduce_min=MagicMock(side_effect=lambda value: value),
        should_send_slot_mapping_to_remote=False,
    )
    connector.kv_manager = MagicMock(server_client_mode=True) if leader else None
    connector._pending_retrieves = []
    connector._staged_retrieves = {}
    for name in (
        "_pending_lookups", "_pending_lookup_contexts", "_prefetch_contexts",
        "_ongoing_prefetches", "_inflight_loads", "_launched_load_contexts",
        "_inflight_stores", "_inflight_store_contexts",
    ):
        setattr(connector, name, {})
    connector._completed_layerwise = []
    connector._launched_load_tids = []
    connector.layer_done_counter = None
    connector._restore_slot_mappings = MagicMock(side_effect=lambda slots: (slots, None))
    connector._new_op_context = MagicMock(side_effect=lambda *a, **kw: SimpleNamespace(
        task_id=-1, operation="lookup",
    ))
    connector._log_cache_op = MagicMock()
    connector._drain_pending_store_batch = MagicMock()
    connector._flush_layerwise_launch_batch = MagicMock()
    connector._uses_queued_prefetch = MagicMock(return_value=False)
    connector._drain_arrival_lookups = MagicMock()
    connector._reset_store_poll = MagicMock()
    return connector


def _stage(connector, rid="r", task_id=7):
    connector._pending_lookups[rid] = task_id
    return connector.stage_retrieve(rid, torch.arange(4, dtype=torch.int64))


def _response(status):
    from flexkv.common.request import KVResponseStatus

    return SimpleNamespace(status=KVResponseStatus(status))


@pytest.mark.parametrize("defer", [False, True])
def test_staged_launch_is_independent_of_ordinary_deferral(defer):
    connector = _staged_connector(defer=defer)

    def launch(**kwargs):
        entry = connector._staged_retrieves["r"]
        assert not entry.launch_confirmed and entry.completely
        assert not connector._pending_retrieves
        assert kwargs["task_ids"] == [7] and kwargs["layerwise_transfer"] is False

    connector.kv_manager.launch.side_effect = launch
    assert _stage(connector)
    assert connector._staged_retrieves["r"].launch_confirmed
    assert not connector._pending_lookups and not connector._pending_lookup_contexts
    connector.kv_manager.wait.assert_not_called()
    assert connector.flush_pending_retrieves() == 0
    connector.kv_manager.wait.assert_not_called()


def test_staged_missing_lookup_and_duplicate_do_not_launch_again():
    connector = _staged_connector()
    assert not connector.stage_retrieve("missing", torch.arange(4))
    connector.kv_manager.launch.assert_not_called()
    assert _stage(connector)
    with pytest.raises(RuntimeError, match="duplicated"):
        connector.stage_retrieve("r", torch.arange(4))
    assert connector.kv_manager.launch.call_count == 1
    assert set(connector._staged_retrieves) == {"r"}


@pytest.mark.parametrize("status, expected", [("success", True), ("failed", False), ("cancelled", False)])
def test_staged_poll_returns_only_fully_drained_terminal_tasks(status, expected):
    connector = _staged_connector()
    assert _stage(connector)
    drained = [False]

    def wait(task_ids, *, timeout, completely):
        assert task_ids == [7] and timeout == 0.0 and completely is True
        return {7: _response(status if drained[0] else "timeout")}

    connector.kv_manager.wait.side_effect = wait
    assert connector.poll_staged_retrieves(["absent", "r"]) == {}
    assert "r" in connector._staged_retrieves
    drained[0] = True
    assert connector.poll_staged_retrieves(["r", "r"]) == {"r": expected}
    assert not connector._staged_retrieves
    assert not connector._pending_retrieves
    connector.kv_manager.cancel.assert_not_called()


def test_staged_poll_missing_response_preserves_task():
    connector = _staged_connector()
    assert _stage(connector)
    connector.kv_manager.wait.return_value = {}
    assert connector.poll_staged_retrieves(["r"]) == {}
    assert "r" in connector._staged_retrieves


@pytest.mark.parametrize("status", ["not_found", "unready"])
def test_staged_poll_does_not_treat_missing_task_as_a_drain_proof(status):
    connector = _staged_connector()
    assert _stage(connector)
    connector.kv_manager.wait.return_value = {7: _response(status)}
    with pytest.raises(RuntimeError, match="drain unproven"):
        connector.poll_staged_retrieves(["r"])
    assert "r" in connector._staged_retrieves


def test_adopt_staged_restore_waits_at_seal_and_does_not_wait_other_staged_tasks():
    connector = _staged_connector()
    assert _stage(connector, "r", 7)
    assert _stage(connector, "later", 8)
    assert connector.adopt_staged_retrieve("r")
    assert not connector.adopt_staged_retrieve("r")
    connector.kv_manager.wait.return_value = {7: _response("success")}
    assert connector.flush_pending_retrieves() == 0
    connector.kv_manager.wait.assert_called_once_with([7], timeout=30.0, completely=True)
    assert not connector._pending_retrieves
    assert set(connector._staged_retrieves) == {"later"}


def test_polled_success_has_no_outstanding_task_to_adopt_or_wait():
    connector = _staged_connector()
    assert _stage(connector)
    connector.kv_manager.wait.return_value = {7: _response("success")}
    assert connector.poll_staged_retrieves(["r"]) == {"r": True}
    assert not connector.adopt_staged_retrieve("r")
    assert connector.flush_pending_retrieves() == 0
    assert connector.kv_manager.wait.call_count == 1


@pytest.mark.parametrize("failure", ["timeout", "exception"])
def test_adopted_seal_failure_retains_task_for_reset(failure):
    connector = _staged_connector()
    assert _stage(connector)
    assert connector.adopt_staged_retrieve("r")
    if failure == "exception":
        connector.kv_manager.wait.side_effect = RuntimeError("wait lost")
    else:
        connector.kv_manager.wait.return_value = {7: _response("timeout")}
    with pytest.raises(RuntimeError):
        connector.flush_pending_retrieves()
    assert [entry.rid for entry in connector._pending_retrieves] == ["r"]
    connector.kv_manager.wait.side_effect = None
    connector.kv_manager.wait.return_value = {7: _response("success")}
    connector.reset()
    assert not connector._pending_retrieves


@pytest.mark.parametrize("status", ["success", "failed", "cancelled"])
def test_reset_drains_staged_before_any_source_cleanup(status):
    connector = _staged_connector()
    assert _stage(connector)

    def wait(task_ids, *, timeout, completely):
        assert task_ids == [7] and timeout == 30.0 and completely is True
        connector._drain_pending_store_batch.assert_not_called()
        return {7: _response(status)}

    connector.kv_manager.wait.side_effect = wait
    connector.reset()
    assert not connector._staged_retrieves
    connector._drain_pending_store_batch.assert_called_once()
    connector.kv_manager.cancel.assert_not_called()


@pytest.mark.parametrize("failure", ["timeout", "exception", "not_found", "unready"])
def test_staged_reset_failure_preserves_ownership_and_stops_cleanup(failure):
    connector = _staged_connector()
    assert _stage(connector)
    entry = connector._staged_retrieves["r"]
    if failure == "exception":
        connector.kv_manager.wait.side_effect = RuntimeError("wait lost")
    else:
        connector.kv_manager.wait.return_value = {7: _response(failure)}
    with pytest.raises(RuntimeError):
        connector.reset()
    assert connector._staged_retrieves == {"r": entry}
    connector._drain_pending_store_batch.assert_not_called()
    connector._drain_arrival_lookups.assert_not_called()


def test_partial_reset_caches_consumed_terminal_proofs_for_retry():
    connector = _staged_connector()
    assert _stage(connector, "done", 7)
    assert _stage(connector, "running", 8)
    connector.kv_manager.wait.return_value = {7: _response("success"), 8: _response("timeout")}
    with pytest.raises(RuntimeError, match="still running"):
        connector.reset()
    assert set(connector._staged_retrieves) == {"done", "running"}
    assert connector._staged_retrieves["done"].terminal_status == "success"
    connector.kv_manager.wait.return_value = {8: _response("success")}
    connector.reset()
    assert connector.kv_manager.wait.call_args.args == ([8],)
    assert not connector._staged_retrieves


def test_launch_exception_is_recorded_before_launch_and_reset_fails_closed():
    connector = _staged_connector()
    connector.kv_manager.launch.side_effect = RuntimeError("launch uncertain")
    with pytest.raises(RuntimeError, match="launch uncertain"):
        _stage(connector)
    entry = connector._staged_retrieves["r"]
    assert not entry.launch_confirmed
    assert not connector._pending_lookups
    connector.kv_manager.wait.return_value = {7: _response("unready")}
    with pytest.raises(RuntimeError, match="drain unproven"):
        connector.reset()
    assert connector._staged_retrieves["r"] is entry
    connector.kv_manager.wait.return_value = {7: _response("success")}
    connector.reset()
    assert not connector._staged_retrieves


def test_mapping_exception_preserves_staged_task_before_launch():
    connector = _staged_connector()
    connector._restore_slot_mappings.side_effect = RuntimeError("mapping failed")
    with pytest.raises(RuntimeError, match="mapping failed"):
        _stage(connector)
    assert "r" in connector._staged_retrieves
    assert not connector._staged_retrieves["r"].launch_confirmed
    connector.kv_manager.launch.assert_not_called()


class _RetrieveMailbox:
    def __init__(self):
        from collections import deque

        self.messages = deque()

    def leader(self, payload, *, channel, blocking=False):
        import copy

        self.messages.append((channel, blocking, copy.deepcopy(payload)))
        return payload

    def follower(self, payload, *, channel, blocking=False):
        expected_channel, expected_blocking, result = self.messages.popleft()
        assert (channel, blocking) == (expected_channel, expected_blocking)
        return result


def _staged_pair():
    mailbox = _RetrieveMailbox()
    return (
        _staged_connector(scatter=mailbox.leader),
        _staged_connector(leader=False, scatter=mailbox.follower),
        mailbox,
    )


def test_staged_poll_uses_leader_order_and_verdict_even_with_empty_local_rids():
    leader, follower, mailbox = _staged_pair()
    for rid, task_id in (("a", 7), ("b", 8), ("running", 9)):
        assert _stage(leader, rid, task_id)
        assert _stage(follower, rid, task_id)
    leader.kv_manager.wait.return_value = {
        7: _response("success"), 8: _response("success"), 9: _response("timeout"),
    }
    expected = {"b": True, "a": True}
    assert leader.poll_staged_retrieves(["b", "absent", "a", "running"]) == expected
    assert follower.poll_staged_retrieves([]) == expected
    leader.kv_manager.wait.assert_called_once_with([8, 7, 9], timeout=0.0, completely=True)
    for connector in (leader, follower):
        assert set(connector._staged_retrieves) == {"running"}
    assert not mailbox.messages


@pytest.mark.parametrize("operation", ["poll", "reset"])
def test_wait_exception_is_shared_before_followers_can_retire_staged_tasks(operation):
    leader, follower, mailbox = _staged_pair()
    assert _stage(leader)
    assert _stage(follower)
    leader.kv_manager.wait.side_effect = RuntimeError("owner disconnected")
    for connector in (leader, follower):
        with pytest.raises(RuntimeError, match="owner disconnected"):
            if operation == "poll":
                connector.poll_staged_retrieves(["r"])
            else:
                connector.reset()
        assert "r" in connector._staged_retrieves
        connector._drain_pending_store_batch.assert_not_called()
    assert not mailbox.messages


def test_launch_exception_is_shared_with_follower_and_both_keep_unknown_state():
    leader, follower, mailbox = _staged_pair()
    leader.kv_manager.launch.side_effect = RuntimeError("launch uncertain")
    for connector in (leader, follower):
        with pytest.raises(RuntimeError, match="launch uncertain"):
            _stage(connector)
        assert "r" in connector._staged_retrieves
        assert not connector._staged_retrieves["r"].launch_confirmed
    assert not mailbox.messages


@pytest.mark.parametrize("operation", ["poll", "reset"])
@pytest.mark.parametrize("unsupported", ["layerwise_enabled", "layerwise_entry", "non_server"])
def test_unsupported_terminal_failures_are_rank_symmetric_fail_closed(operation, unsupported):
    leader, follower, mailbox = _staged_pair()
    for rid, task_id in (("cancelled", 7), ("failed", 8)):
        assert _stage(leader, rid, task_id)
        assert _stage(follower, rid, task_id)
    for connector in (leader, follower):
        if unsupported == "layerwise_enabled":
            connector.enable_layerwise = True
        elif unsupported == "layerwise_entry":
            for entry in connector._staged_retrieves.values():
                entry.transfer_mode = "layerwise"
    if unsupported == "non_server":
        leader.kv_manager.server_client_mode = False
    leader.kv_manager.wait.return_value = {7: _response("cancelled"), 8: _response("failed")}
    for connector in (leader, follower):
        with pytest.raises(RuntimeError, match="drain unproven"):
            if operation == "poll":
                connector.poll_staged_retrieves(["cancelled", "failed"])
            else:
                connector.reset()
        assert set(connector._staged_retrieves) == {"cancelled", "failed"}
        connector._drain_pending_store_batch.assert_not_called()
    assert not mailbox.messages


@pytest.mark.parametrize("operation", ["poll", "reset"])
def test_fenced_terminal_failure_and_cancel_have_rank_symmetric_cleanup(operation):
    leader, follower, mailbox = _staged_pair()
    for rid, task_id in (("cancelled", 7), ("failed", 8)):
        assert _stage(leader, rid, task_id)
        assert _stage(follower, rid, task_id)
    leader.kv_manager.wait.return_value = {7: _response("cancelled"), 8: _response("failed")}
    for connector in (leader, follower):
        if operation == "poll":
            assert connector.poll_staged_retrieves(["cancelled", "failed"]) == {
                "cancelled": False, "failed": False,
            }
        else:
            connector.reset()
        assert not connector._staged_retrieves
    assert not mailbox.messages


def test_poll_ownership_mismatch_fails_closed_before_wait():
    connector = _staged_connector()
    assert _stage(connector)
    connector._sync_ctx.needs_sync = True
    connector._sync_ctx.scatter = MagicMock(return_value={"entries": [("r", 999)]})
    connector._sync_ctx.all_reduce_min.return_value = 0
    connector._sync_ctx.all_reduce_min.side_effect = None
    with pytest.raises(RuntimeError, match="ownership differs"):
        connector.poll_staged_retrieves(["r"])
    connector.kv_manager.wait.assert_not_called()
    assert "r" in connector._staged_retrieves


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED", "CANCELLED"])
def test_staged_poll_with_real_task_wait_ignores_early_end_until_full_graph_drains(terminal):
    import numpy as np

    from flexkv.kvtask import KVTask, KVTaskEngine, TaskStatus, TaskType

    connector = _staged_connector()
    assert _stage(connector)
    task = KVTask(
        task_id=7, task_type=TaskType.GET, task_end_op_id=1,
        task_end_op_finished=True, status=TaskStatus.RUNNING,
        token_ids=np.arange(4), slot_mapping=np.arange(4), token_mask=None,
        graph=SimpleNamespace(graph_id=10, num_ops=2), return_mask=np.ones(4, dtype=bool),
        callback=None, op_callback_dict={}, cancel_requested=terminal == "CANCELLED",
    )
    engine = KVTaskEngine.__new__(KVTaskEngine)
    engine.tasks = {7: task}
    engine._update_tasks = MagicMock()
    engine._release_task = MagicMock(side_effect=lambda tid: engine.tasks.pop(tid))
    connector.kv_manager.wait.side_effect = lambda task_ids, timeout, completely: engine._wait_impl(
        task_ids, timeout=timeout, completely=completely,
    )

    assert connector.poll_staged_retrieves(["r"]) == {}
    assert "r" in connector._staged_retrieves and task.request_returned is False
    engine._release_task.assert_not_called()
    task.status = getattr(TaskStatus, terminal)
    engine.tracer = SimpleNamespace(trace_wait_request=MagicMock())
    assert engine.try_wait([]) == {}
    assert task.request_returned is False
    engine._release_task.assert_not_called()
    assert connector.poll_staged_retrieves(["r"]) == {"r": terminal == "COMPLETED"}
    engine._release_task.assert_called_once_with(7)
    assert not engine.tasks and not connector._staged_retrieves


def test_staged_reset_timeout_is_shared_before_followers_can_free():
    leader, follower, mailbox = _staged_pair()
    assert _stage(leader)
    assert _stage(follower)
    leader.kv_manager.wait.return_value = {7: _response("timeout")}
    for connector in (leader, follower):
        with pytest.raises(RuntimeError, match="still running"):
            connector.reset()
        assert "r" in connector._staged_retrieves
        connector._drain_pending_store_batch.assert_not_called()
    assert not mailbox.messages
