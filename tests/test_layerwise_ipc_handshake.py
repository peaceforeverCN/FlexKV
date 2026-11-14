"""The extension's IPC handle export must round-trip into a usable event.

test_ipc_layer_event_crossproc.py proves CUDA interprocess events work and
that the consumer's half is correct, but it simulates the producer with
ctypes. This closes the remaining gap: the handles the *extension* exports
(``RegionBatchGroup.init_ipc_events`` / ``export_ipc_handles``, which is what
the worker actually calls during the eventfd handshake) must be adoptable by
the consumer object, in the right shape and the right axis order.

Getting the axis order wrong is the failure this is really guarding against:
the grid is [counter][rank][layer] to match the eventfd table, and a
transposed export would still produce valid, openable handles -- just ones
belonging to the wrong layer. Nothing downstream would complain; forward
would simply wait on the wrong marker.

Run (needs one GPU):
    python tests/test_layerwise_ipc_handshake.py
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HANDLE_SIZE = 64
NUM_COUNTERS = 3
NUM_LAYERS = 5


def _adopt_in_child(conn, device: int, sliced: bytes) -> None:
    """Adopt the parent's exported handles, as the sglang rank would."""
    try:
        import torch

        from flexkv.integration.sglang.comm import FlexKVLayerDoneCounter

        torch.cuda.set_device(device)
        counter = FlexKVLayerDoneCounter(
            num_layers=NUM_LAYERS, num_counters=NUM_COUNTERS, world_rank=0)
        counter.attach_ipc_handles(sliced)
        if not counter.pipelined:
            conn.send((False, "counter did not switch to pipelined mode"))
            return

        # The adopted events must actually be able to order a stream -- that
        # they opened only proves the handles were well-formed.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for slot in range(NUM_COUNTERS):
                for layer in range(NUM_LAYERS):
                    counter.events[slot]._stream_wait_layer(layer)
        torch.cuda.synchronize()
        conn.send((
            True,
            f"{NUM_COUNTERS} slots x {NUM_LAYERS} layers adopted and used to "
            "order a stream",
        ))
    except Exception as exc:  # noqa: BLE001
        import traceback

        conn.send((False, f"{exc}\n{traceback.format_exc()}"))


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    from flexkv.c_ext import RegionBatchGroup, RegionDesc
    from flexkv.integration.sglang.comm import (
        FlexKVLayerDoneCounter,
        FlexKVLayerLoadingEvent,
    )
    from flexkv.transfer.layer_eventfd import _slice_ipc_blob

    device = 0
    torch.cuda.set_device(device)
    failures = []

    # A minimally-valid region: the event grid is a property of the device
    # list and the declared shape, not of any transfer, so the region only has
    # to pass construction validation. Nothing is ever submitted against it.
    region = RegionDesc()
    region.name = "probe"
    region.num_layers = NUM_LAYERS
    region.kv_dim = 1
    region.num_kv_heads = 1
    region.num_tensors_per_gpu = 1
    region.cpu_ptr = 0
    region.cpu_block_stride_in_bytes = 1
    region.cpu_kv_stride_in_bytes = 1
    region.cpu_layer_stride_in_bytes = 1
    region.cpu_tp_stride_in_bytes = 1
    region.gpu_block_ptrs_flat = [0]
    region.gpu_block_strides_in_bytes = [1]
    region.gpu_chunk_sizes_in_bytes = [1]
    region.gpu_kv_strides_in_bytes = [1]
    region.gpu_layer_strides_in_bytes = [1]
    batch = RegionBatchGroup([device], [region])

    batch.init_ipc_events(NUM_COUNTERS, NUM_LAYERS)
    blob = batch.export_ipc_handles()

    expected = NUM_COUNTERS * 1 * NUM_LAYERS * HANDLE_SIZE
    if len(blob) != expected:
        print(f"FAIL shape: exported {len(blob)} bytes, expected {expected}")
        return 1
    print(f"PASS shape: exported {len(blob)} bytes for "
          f"[{NUM_COUNTERS},1,{NUM_LAYERS}]")

    # Idempotent re-arm: the handshake can be retried, and a second call with
    # the same shape must not invalidate handles already handed out.
    batch.init_ipc_events(NUM_COUNTERS, NUM_LAYERS)
    if batch.export_ipc_handles() != blob:
        failures.append(
            "re-arming with the same shape changed the handles; a retried "
            "handshake would leave the consumer holding stale ones"
        )
    else:
        print("PASS idempotent: re-arming the same shape is a no-op")

    # Every handle must be distinct. A repeated handle means some (counter,
    # layer) pair shares an event with another, which is exactly the
    # cross-generation collision the counter dimension was added to prevent.
    handles = {
        blob[i * HANDLE_SIZE:(i + 1) * HANDLE_SIZE]
        for i in range(len(blob) // HANDLE_SIZE)
    }
    if len(handles) != NUM_COUNTERS * NUM_LAYERS:
        failures.append(
            f"only {len(handles)} distinct handles for "
            f"{NUM_COUNTERS * NUM_LAYERS} (counter, layer) pairs: slots share "
            "events, so one batch's recording would clobber another's"
        )
    else:
        print(f"PASS distinct: {len(handles)} unique handles")

    # The slice the handshake sends to rank 0 must be adoptable as-is, and
    # must carry num_counters runs of num_layers handles.
    sliced = _slice_ipc_blob(blob, 0, NUM_COUNTERS, 1, NUM_LAYERS)
    if sliced != blob:
        failures.append("single-rank slice should be the whole grid")
    else:
        print("PASS slice: rank 0 of 1 receives the whole grid")

    # Adoption has to happen in another process: CUDA refuses to open an IPC
    # handle in the process that exported it, which is also why the worker is
    # the exporter and sglang the adopter rather than the other way round.
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    adopter = ctx.Process(target=_adopt_in_child, args=(child, device, sliced))
    adopter.start()
    try:
        ok, detail = parent.recv()
    finally:
        adopter.join(timeout=60)
        if adopter.is_alive():
            adopter.terminate()
    if not ok:
        print(f"FAIL adopt: {detail}")
        return 1
    print(f"PASS adopt: {detail}")

    # A reshape must be refused rather than silently re-exported: the
    # consumer's handles are already open at that point.
    try:
        batch.init_ipc_events(NUM_COUNTERS + 1, NUM_LAYERS)
        failures.append(
            "re-arming with a different shape was accepted; the consumer's "
            "open handles would silently refer to destroyed events"
        )
    except Exception:
        print("PASS guarded: re-arming with a different shape is refused")

    # Double-adopting must be refused too (it would leak the first set).
    try:
        FlexKVLayerLoadingEvent(NUM_LAYERS).attach_ipc_events(
            blob[:NUM_LAYERS * HANDLE_SIZE] * 2)
        failures.append("attach_ipc_events accepted a wrongly-sized blob")
    except ValueError:
        print("PASS guarded: a wrongly-sized handle blob is rejected")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"unexpected error for a wrongly-sized blob: {exc}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL PASS: the extension's handle export is consumable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
