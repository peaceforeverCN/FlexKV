"""The layerwise ring must let batch N+1 load while batch N is in forward.

This is the property the pipelined path exists for, and the one the
serialised path structurally cannot have. The tests below are written against
``FlexKVLayerDoneCounter`` alone -- no CUDA, no worker, no sglang -- because
the obstruction was never in the transfer, it was in this state machine:

  * ``expected_transfer_count`` rejected any slot while another was the
    consumer, so arming a second batch raised outright;
  * ``update_producer`` returned the *consumer's* slot when it was still open,
    so a second batch would have silently shared one slot's eventfds;
  * the consumer slot was released only when forward waited for the last
    layer, which can only be known to have happened if that wait blocked.

Each test therefore asserts a difference in behaviour between the two modes
rather than just "pipelined works", so that a regression which quietly
re-serialises the ring fails here instead of only showing up as a benchmark
number.

Run: pytest -q tests/test_layerwise_pipeline_ring.py
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexkv.integration.sglang.comm import (  # noqa: E402
    FlexKVLayerDoneCounter,
    FlexKVLayerLoadingEvent,
)

NUM_LAYERS = 4
NUM_SLOTS = 3


class FakeReleaseEvent:
    """A cudaEvent stand-in whose completion the test drives by hand.

    The real release marker fires when the GPU drains past forward's last
    read of the slot. Tests need that moment to be explicit, because the
    whole question is what the ring does *before* it arrives.
    """

    def __init__(self) -> None:
        self.recorded = False
        self.complete = False

    def record(self) -> None:
        self.recorded = True
        self.complete = False

    def query(self) -> bool:
        return self.complete


class FakePipelinedEvent(FlexKVLayerLoadingEvent):
    """A slot that behaves as pipelined without needing a GPU.

    Only the three CUDA touchpoints are replaced: adopting interprocess
    events, ordering the consumer's stream behind a layer, and the release
    marker. The eventfds are real -- they are cheap, and their semaphore
    semantics are part of what is under test.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__(num_layers)
        # Non-empty => ``pipelined`` is True. Never dereferenced, because
        # _stream_wait_layer is overridden below.
        self._ipc_events = [None] * num_layers  # type: ignore[list-item]
        self.stream_waits: list[int] = []

    def _stream_wait_layer(self, layer_index: int) -> None:
        self.stream_waits.append(layer_index)

    def mark_release_point(self) -> None:
        if self._released_event is None:
            self._released_event = FakeReleaseEvent()  # type: ignore[assignment]
        self._released_event.record()
        self._release_pending = True

    def finish_on_gpu(self) -> None:
        """Pretend the GPU drained past this slot's last read."""
        assert self._released_event is not None, (
            "slot was never marked released; forward did not reach its last layer"
        )
        self._released_event.complete = True  # type: ignore[attr-defined]


def make_counter(pipelined: bool) -> FlexKVLayerDoneCounter:
    counter = FlexKVLayerDoneCounter(
        num_layers=NUM_LAYERS, num_counters=NUM_SLOTS, world_rank=0
    )
    if pipelined:
        for event in counter.events:
            event.close()
        counter.events = [FakePipelinedEvent(NUM_LAYERS) for _ in range(NUM_SLOTS)]
        assert counter.pipelined
    else:
        assert not counter.pipelined
    # Keep the no-slot-available path from hanging the suite.
    counter.wait_timeout_s = 1.0
    return counter


def admit_batch(counter: FlexKVLayerDoneCounter, task_id: int) -> int:
    """One batch's worth of admission, as the connector drives it."""
    slot = counter.update_producer()
    counter.register_task(task_id, slot)
    counter.set_consumer(task_id)
    return counter.seal_producer()


def run_forward(
    counter: FlexKVLayerDoneCounter, slot: int, *, post_signals: bool = True
) -> None:
    """One batch's worth of forward: post the worker's fds, then wait layers."""
    event = counter.events[slot]
    if post_signals:
        pending = max(1, len(counter._slot_task_ids.get(slot, [])) or 1)
        for layer in range(NUM_LAYERS):
            for _ in range(pending):
                os.write(event.load_event_fds[layer], (1).to_bytes(8, sys.byteorder))
    counter.activate_consumer(slot)
    for layer in range(NUM_LAYERS):
        counter.wait_until(layer)


# ----------------------------------------------------------------------
# The core property
# ----------------------------------------------------------------------


def test_second_batch_can_arm_while_first_is_unconsumed_pipelined():
    """Batch N+1 gets its own slot while batch N has not run forward yet.

    This is two-batch, reduced to its smallest observable form.
    """
    counter = make_counter(pipelined=True)

    slot_a = admit_batch(counter, task_id=1)
    assert slot_a >= 0

    # Nothing has consumed slot A: no activate_consumer, no waits.
    slot_b = admit_batch(counter, task_id=2)

    assert slot_b >= 0
    assert slot_b != slot_a, (
        "batch N+1 reused batch N's slot; their eventfd signals would be "
        "indistinguishable and forward would read the wrong generation"
    )


def test_second_batch_is_refused_while_first_is_unconsumed_serialised():
    """The same sequence on the serialised path, to pin down what changed.

    Not an aspiration -- this is the pre-existing behaviour, and it is what
    made the pipeline impossible rather than merely slow.
    """
    counter = make_counter(pipelined=False)

    slot_a = admit_batch(counter, task_id=1)
    assert slot_a >= 0

    # update_producer hands back the consumer's own slot, so the second batch
    # does not get a slot of its own at all...
    slot_again = counter.update_producer()
    assert slot_again == counter.consumer_index

    # ...and asking for any other slot is rejected outright.
    other = (slot_again + 1) % NUM_SLOTS
    with pytest.raises(RuntimeError, match="still the active consumer"):
        counter.expected_transfer_count(other)


def test_three_batches_in_flight_then_backpressure():
    """The ring bounds how far ahead the scheduler may run, and says so.

    Three slots means three live batches; a fourth must wait rather than
    overwrite a slot that forward is still reading. Degrading to a wait is the
    point -- silently reusing the slot would corrupt KV.
    """
    counter = make_counter(pipelined=True)

    slots = [admit_batch(counter, task_id=i) for i in range(NUM_SLOTS)]
    assert len(set(slots)) == NUM_SLOTS, f"slots must be distinct, got {slots}"

    with pytest.raises(TimeoutError, match="became reusable"):
        admit_batch(counter, task_id=99)


def test_slot_returns_to_ring_only_after_gpu_drains():
    """A slot is reusable when the GPU is done with it, not when forward queued it.

    Forward reaching its last layer only means the reads are *issued*. If the
    ring accepted that as release, batch N+3's H2D could overwrite pages batch
    N's attention had not read yet.
    """
    counter = make_counter(pipelined=True)
    slots = [admit_batch(counter, task_id=i) for i in range(NUM_SLOTS)]

    run_forward(counter, slots[0])

    event = counter.events[slots[0]]
    assert event._finished, "last layer's wait should close the slot's batch"
    assert not event.is_released(), (
        "slot was offered back to the ring while its reads were still queued"
    )

    with pytest.raises(TimeoutError):
        admit_batch(counter, task_id=100)

    event.finish_on_gpu()
    assert event.is_released()

    reused = admit_batch(counter, task_id=101)
    assert reused == slots[0], "the drained slot should be the one reused"


# ----------------------------------------------------------------------
# Correctness of the split signal (fd = generation, event = data)
# ----------------------------------------------------------------------


def test_forward_waits_on_its_own_batchs_slot_not_the_newest():
    """Forward must wait on the slot its batch loaded into.

    The regression this guards against is the reason the slot is carried on
    the batch at all: with a single global "current consumer", admitting batch
    N+1 before batch N's forward would re-point the wait at N+1's slot, and
    batch N would proceed on data that was never loaded for it.
    """
    counter = make_counter(pipelined=True)

    slot_a = admit_batch(counter, task_id=1)
    slot_b = admit_batch(counter, task_id=2)
    assert slot_a != slot_b

    run_forward(counter, slot_a)

    assert counter.events[slot_a].stream_waits == list(range(NUM_LAYERS))
    assert counter.events[slot_b].stream_waits == [], (
        "forward of batch N ordered itself behind batch N+1's transfer"
    )


def test_every_layer_is_ordered_on_the_gpu():
    """Each layer's wait must reach the GPU, not just consume an fd.

    The fd only proves the copies are enqueued. If a layer's
    cudaStreamWaitEvent were skipped, forward would read that layer before its
    H2D landed -- a silent corruption that no eventfd accounting would catch.
    """
    counter = make_counter(pipelined=True)
    slot = admit_batch(counter, task_id=1)

    run_forward(counter, slot)

    assert counter.events[slot].stream_waits == list(range(NUM_LAYERS))


def test_fd_signals_are_still_consumed_once_per_transfer():
    """Pipelining must not break the semaphore accounting.

    Several restores can share a batch, and each posts its own token per
    layer. Leaving tokens unread would make the *next* transfer on that slot
    return immediately from a stale signal.
    """
    counter = make_counter(pipelined=True)

    slot = counter.update_producer()
    for task_id in (1, 2, 3):
        counter.register_task(task_id, slot)
        counter.set_consumer(task_id)
    assert counter.seal_producer() == slot
    assert counter.events[slot].pending_transfers() == 3

    run_forward(counter, slot)

    event = counter.events[slot]
    for layer, fd in enumerate(event.load_event_fds):
        with pytest.raises(BlockingIOError):
            os.read(fd, 8)
        assert event.wait_remaining[layer] == 0


def test_rollback_returns_an_open_slot_to_the_ring():
    """A failed preflight must not strand the slot it had selected.

    With three slots, leaking one per failure takes the ring down after three
    failures -- and the failure path is exactly where that is least welcome.
    """
    counter = make_counter(pipelined=True)

    slot = counter.update_producer()
    counter.register_task(7, slot)
    counter.set_consumer(7)
    counter.rollback_prepared_transfer(7, slot, transfer_added=True)

    assert counter.events[slot].is_released()
    assert counter._open_producer == -1

    # The same slot is immediately selectable again.
    slots = [admit_batch(counter, task_id=i) for i in range(NUM_SLOTS)]
    assert slot in slots
    assert len(set(slots)) == NUM_SLOTS


def test_follower_adopts_the_leaders_slot_for_a_shared_batch():
    """Followers never call update_producer; they are told the slot.

    A second restore joining a follower's batch must be accepted. Before the
    pipelined path tracked an "open producer" on followers too, this raised
    "slot in use" on every rank but the leader -- which only shows up with
    more than one restore per batch.
    """
    counter = make_counter(pipelined=True)
    leader_slot = 0

    counter.register_task_with_explicit_counter_id(1, leader_slot)
    counter.set_consumer(1)

    # Second restore in the same batch, as the leader broadcast it.
    assert counter.expected_transfer_count(leader_slot) == 2
    counter.register_task_with_explicit_counter_id(2, leader_slot)
    counter.set_consumer(2)

    assert counter.seal_producer() == leader_slot
    assert counter.events[leader_slot].pending_transfers() == 2


def test_empty_batch_gives_its_slot_straight_back():
    """Sealing a batch that loaded nothing must not consume a slot."""
    counter = make_counter(pipelined=True)

    counter.update_producer()
    assert counter.seal_producer() == -1

    slots = [admit_batch(counter, task_id=i) for i in range(NUM_SLOTS)]
    assert len(set(slots)) == NUM_SLOTS, "sealing an empty batch leaked a slot"


def test_unwired_scheduler_is_an_error_not_a_silent_skip():
    """A sealed slot that no forward claims must raise, not be ignored.

    This is a regression test for a real, nearly-shipped bug. The pipelined
    path stops ``set_consumer`` from setting ``consumer_index`` -- only
    ``activate_consumer`` does, driven by the scheduler carrying the slot on
    the batch. The first integration attempt wired that into
    ``FlexKVRadixCache`` while the deployment actually used
    ``FlexKVHybridRadixCache``, so it was never called.

    The symptom was not a crash: ``wait_until`` saw ``consumer_index < 0``,
    returned immediately for all 40 layers, and forward read KV whose H2D had
    not landed. A smoke test still answered "1+1=2" and a throughput
    benchmark still produced plausible numbers.
    """
    counter = make_counter(pipelined=True)
    slot = admit_batch(counter, task_id=1)
    assert slot >= 0

    # Forward begins without the scheduler ever naming the slot.
    with pytest.raises(RuntimeError, match="never told which slot"):
        counter.wait_until(0)


def test_wired_scheduler_passes_the_guard():
    """The guard must not fire on the correct sequence."""
    counter = make_counter(pipelined=True)
    slot = admit_batch(counter, task_id=1)

    run_forward(counter, slot)  # activate_consumer, then all layers

    assert counter._pending_activation == -1


def test_guard_is_quiet_for_a_batch_that_loaded_nothing():
    """A batch with no restores has no slot, and that is not an error."""
    counter = make_counter(pipelined=True)

    counter.update_producer()
    assert counter.seal_producer() == -1
    counter.activate_consumer(-1)

    for layer in range(NUM_LAYERS):
        counter.wait_until(layer)  # must not raise


def test_guard_is_quiet_after_rollback():
    """A slot released by a failed preflight must not trip the guard."""
    counter = make_counter(pipelined=True)

    slot = counter.update_producer()
    counter.register_task(5, slot)
    counter.set_consumer(5)
    counter.seal_producer()
    counter.rollback_prepared_transfer(5, slot, transfer_added=True)

    counter.wait_until(0)  # must not raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
