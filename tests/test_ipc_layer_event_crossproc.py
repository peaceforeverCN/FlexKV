"""Can a cudaEvent recorded in another process order sglang's forward stream?

The pipelined layerwise path rests on this and nothing else: the FlexKV
worker records a marker per layer on its own stream, in its own process and
its own CUDA context, and the consumer orders its forward stream behind that
marker instead of blocking a thread on an eventfd. If interprocess events do
not survive that crossing, the design is void -- so it is checked directly.

The consumer side is the real code: ``FlexKVLayerLoadingEvent`` and the
ctypes shim the connector uses. The producer side is simulated with ctypes
rather than driven through the extension, because what needs proving here is
the *crossing* and the consumer's half of it; the extension's export path is
covered by the handshake in test_layerwise_ipc_handshake.py.

Three properties, in increasing strength:

  1. handles exported by one process can be adopted by another;
  2. waiting on one really orders work -- the consumer's stream cannot drain
     past the wait while the producer's transfer is still running;
  3. issuing the waits costs the consumer's CPU nothing, which is the whole
     reason the scheduler can run a batch ahead.

Run (needs one GPU):
    python tests/test_ipc_layer_event_crossproc.py
"""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_LAYERS = 4
HANDLE_SIZE = 64

cudaEventDisableTiming = 0x02
cudaEventInterprocess = 0x04

# Large enough that the H2D is unambiguously still running when the consumer
# queues its wait; that overlap is the situation being tested.
ELEMS = 96 * 1024 * 1024


def _bind(lib):
    from flexkv.integration.sglang.comm import _CudaIpcEventHandle

    lib.cudaEventCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cudaEventCreateWithFlags.restype = ctypes.c_int
    lib.cudaIpcGetEventHandle.argtypes = [
        ctypes.POINTER(_CudaIpcEventHandle), ctypes.c_void_p]
    lib.cudaIpcGetEventHandle.restype = ctypes.c_int
    lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventRecord.restype = ctypes.c_int
    return _CudaIpcEventHandle


def producer(conn, device: int) -> None:
    """Stand in for the FlexKV worker process."""
    try:
        import torch

        from flexkv.integration.sglang.comm import _cuda_check, _load_cudart

        torch.cuda.set_device(device)
        lib = _load_cudart()
        HandleT = _bind(lib)

        events = []
        blob = bytearray()
        for _ in range(NUM_LAYERS):
            event = ctypes.c_void_p()
            _cuda_check(
                lib.cudaEventCreateWithFlags(
                    ctypes.byref(event),
                    cudaEventInterprocess | cudaEventDisableTiming,
                ),
                "cudaEventCreateWithFlags",
            )
            handle = HandleT()
            _cuda_check(
                lib.cudaIpcGetEventHandle(ctypes.byref(handle), event),
                "cudaIpcGetEventHandle",
            )
            events.append(event)
            # string_at on the address, not handle.reserved: accessing a
            # c_char array as an attribute stops at the first NUL, and these
            # handles are binary -- that truncation silently yields a few
            # bytes per handle instead of 64.
            blob += ctypes.string_at(ctypes.addressof(handle), HANDLE_SIZE)

        conn.send(("handles", bytes(blob)))
        assert conn.recv() == "go"

        stream = torch.cuda.Stream()
        src = torch.ones(ELEMS, dtype=torch.float32).pin_memory()
        dst = torch.zeros(ELEMS, dtype=torch.float32, device=device)

        with torch.cuda.stream(stream):
            for layer in range(NUM_LAYERS):
                dst.copy_(src, non_blocking=True)
                _cuda_check(
                    lib.cudaEventRecord(
                        events[layer], ctypes.c_void_p(stream.cuda_stream)),
                    f"cudaEventRecord(layer={layer})",
                )
        # Everything is on the stream; nothing has necessarily landed.
        conn.send(("recorded", time.monotonic()))
        stream.synchronize()
        conn.send(("landed", time.monotonic()))
        assert conn.recv() == "done"
        conn.send(("ok", None))
    except Exception as exc:  # noqa: BLE001
        import traceback

        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    device = 0
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=producer, args=(child, device))
    proc.start()
    failures = []

    try:
        kind, payload = parent.recv()
        if kind == "error":
            print(f"FAIL producer: {payload}")
            return 1
        assert kind == "handles", kind

        torch.cuda.set_device(device)
        from flexkv.integration.sglang.comm import FlexKVLayerLoadingEvent

        # --- 1. adoption, through the real consumer-side object -------------
        event = FlexKVLayerLoadingEvent(NUM_LAYERS)
        event.attach_ipc_events(payload)
        assert event.pipelined
        print(f"PASS adopt: opened {NUM_LAYERS} interprocess events")

        parent.send("go")
        kind, _ = parent.recv()
        assert kind == "recorded", kind

        # --- 2./3. order a stream behind them, and time the CPU cost -------
        consumer_stream = torch.cuda.Stream()
        with torch.cuda.stream(consumer_stream):
            t0 = time.monotonic()
            for layer in range(NUM_LAYERS):
                event._stream_wait_layer(layer)
            cpu_cost = time.monotonic() - t0
            after_waits = torch.cuda.Event()
            after_waits.record()

        # Queued behind four waits on a transfer that is still running, so
        # this must not be complete yet. If it is, the waits ordered nothing.
        drained_early = after_waits.query()

        if cpu_cost > 0.010:
            failures.append(
                f"queuing {NUM_LAYERS} waits cost {cpu_cost * 1e3:.2f} ms of "
                "CPU; the consumer should not wait for data on its CPU"
            )
        else:
            print(
                f"PASS non-blocking: queued {NUM_LAYERS} waits in "
                f"{cpu_cost * 1e3:.3f} ms of CPU time"
            )

        if drained_early:
            failures.append(
                "the consumer's stream drained past the waits while the "
                "producer's transfer was still in flight: the interprocess "
                "event did not order anything"
            )
        else:
            print("PASS ordering: consumer's stream is held behind the transfer")

        kind, _ = parent.recv()
        assert kind == "landed", kind

        after_waits.synchronize()
        if not after_waits.query():
            failures.append("consumer's stream never drained after the transfer landed")
        else:
            print("PASS release: consumer's stream drained once the transfer landed")

        parent.send("done")
        kind, payload = parent.recv()
        if kind == "error":
            print(f"FAIL producer: {payload}")
            return 1

        event.close()
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL PASS: interprocess layer events cross the process boundary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
