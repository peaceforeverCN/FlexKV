"""The eventfd handshake must stay compatible in both directions.

The pipelined path adds a reply to a handshake that previously ended at a
one-byte ACK. That is an easy thing to get subtly wrong, and the failure is
not a crash -- it is a 30-second socket timeout during startup, on the
*default* path, for users who never opted in.

Two directions matter:

  * new worker + old consumer: the worker now sends ACK plus a length. An
    old consumer reads only the ACK, leaves the rest in the socket buffer,
    and must be unaffected.
  * new consumer + old worker: the worker sends a bare ACK. A new consumer
    must not block reading a length that will never come -- which is why it
    only reads one when FLEXKV_LAYERWISE_PIPELINE is actually set.

Exercised over a real UDS against the real server loop
(``receive_layer_eventfds``), because this is a protocol question and a mock
would just encode whatever I assumed.

Run: pytest -q tests/test_layerwise_handshake_compat.py
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexkv.integration.sglang.comm import (  # noqa: E402
    EFD_NONBLOCK,
    EFD_SEMAPHORE,
    eventfd,
    recv_exactly,
    send_fds,
)
from flexkv.transfer.layer_eventfd import (  # noqa: E402
    IPC_EVENT_HANDLE_SIZE,
    _slice_ipc_blob,
    receive_layer_eventfds,
)

NUM_LAYERS = 3
NUM_COUNTERS = 2
TP_SIZE = 1


@pytest.fixture
def sock_path(tmp_path):
    return str(tmp_path / "lw.sock")


def run_server(sock_path: str, blob: bytes, out: dict) -> threading.Thread:
    """The worker's half, in a thread."""

    def body():
        try:
            out["tensor"] = receive_layer_eventfds(
                sock_path, TP_SIZE, NUM_LAYERS,
                max_retries=10, retry_interval=0.2,
                on_shape_known=lambda nc, nl: blob,
            )
        except Exception as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target=body, daemon=True)
    t.start()
    return t


def client_send_fds(sock_path: str, timeout: float = 20.0) -> socket.socket:
    """The consumer's half up to and including the ACK."""
    deadline = __import__("time").monotonic() + timeout
    sock = None
    while __import__("time").monotonic() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(sock_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            sock = None
            __import__("time").sleep(0.1)
    assert sock is not None, "server never came up"

    sock.sendall(struct.pack("iiii", 0, TP_SIZE, NUM_LAYERS, NUM_COUNTERS))
    for counter_id in range(NUM_COUNTERS):
        fds = [eventfd(0, EFD_SEMAPHORE | EFD_NONBLOCK) for _ in range(NUM_LAYERS)]
        try:
            send_fds(sock, fds, struct.pack("i", counter_id))
        finally:
            for fd in fds:
                os.close(fd)
    sock.settimeout(20.0)
    ack = sock.recv(1)
    assert ack and ack[0] == 1, f"expected ACK, got {ack!r}"
    return sock


def test_old_consumer_ignores_the_handle_reply(sock_path):
    """An old consumer stops at the ACK, and the extra bytes are harmless.

    This is the upgrade-the-worker-first case. The length prefix exists so
    that the worker does not have to know which kind of consumer it is
    talking to.
    """
    total = NUM_COUNTERS * TP_SIZE * NUM_LAYERS * IPC_EVENT_HANDLE_SIZE
    blob = bytes(i % 256 for i in range(total))
    assert len(blob) == total
    out: dict = {}
    t = run_server(sock_path, blob, out)

    sock = client_send_fds(sock_path)
    # Old consumer: never reads past the ACK.
    sock.close()

    t.join(timeout=30)
    assert "error" not in out, out.get("error")
    assert out["tensor"].numel() == NUM_COUNTERS * TP_SIZE * NUM_LAYERS, (
        "the server did not complete the handshake for a consumer that "
        "ignored the handle reply"
    )


def test_new_consumer_reads_its_own_slice(sock_path):
    """A pipelined consumer gets exactly its rank's handles, intact."""
    total = NUM_COUNTERS * TP_SIZE * NUM_LAYERS * IPC_EVENT_HANDLE_SIZE
    # Distinctive per-handle content so a mis-slice is visible.
    blob = b"".join(
        bytes([i % 256]) * IPC_EVENT_HANDLE_SIZE
        for i in range(total // IPC_EVENT_HANDLE_SIZE)
    )
    out: dict = {}
    t = run_server(sock_path, blob, out)

    sock = client_send_fds(sock_path)
    (blob_len,) = struct.unpack("i", recv_exactly(sock, 4))
    assert blob_len == NUM_COUNTERS * NUM_LAYERS * IPC_EVENT_HANDLE_SIZE, (
        f"expected this rank's {NUM_COUNTERS}x{NUM_LAYERS} handles, "
        f"got {blob_len} bytes"
    )
    received = recv_exactly(sock, blob_len)
    sock.close()
    t.join(timeout=30)
    assert "error" not in out, out.get("error")

    expected = _slice_ipc_blob(blob, 0, NUM_COUNTERS, TP_SIZE, NUM_LAYERS)
    assert received == expected, "the rank received the wrong slice of the grid"


def test_non_pipelined_worker_replies_zero_length(sock_path):
    """With no handles, the reply is a length of 0 -- not a missing field.

    A new consumer that has pipelining off does not read this, but a worker
    cannot know that, so the field must always be present and well-formed.
    """
    out: dict = {}
    t = run_server(sock_path, b"", out)

    sock = client_send_fds(sock_path)
    (blob_len,) = struct.unpack("i", recv_exactly(sock, 4))
    sock.close()
    t.join(timeout=30)

    assert "error" not in out, out.get("error")
    assert blob_len == 0, f"expected a zero-length reply, got {blob_len}"


def test_slice_is_per_rank_not_contiguous():
    """A rank's handles are num_counters separate runs, not one block.

    The grid is [counter][rank][layer] to match the eventfd table. Treating a
    rank's share as one contiguous slice would hand it other ranks' handles
    for counters 1+ -- openable, wrong, and silent.
    """
    tp_size = 4
    num_counters = 3
    num_layers = 2
    per = IPC_EVENT_HANDLE_SIZE
    total = num_counters * tp_size * num_layers
    blob = b"".join(bytes([i]) * per for i in range(total))

    for rank in range(tp_size):
        got = _slice_ipc_blob(blob, rank, num_counters, tp_size, num_layers)
        assert len(got) == num_counters * num_layers * per
        # Reconstruct which grid entries we expect for this rank.
        want_ids = [
            (counter * tp_size + rank) * num_layers + layer
            for counter in range(num_counters)
            for layer in range(num_layers)
        ]
        got_ids = [got[i * per] for i in range(len(got) // per)]
        assert got_ids == want_ids, (
            f"rank {rank} got grid entries {got_ids}, expected {want_ids}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
