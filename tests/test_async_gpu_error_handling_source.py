"""Source-level contracts for multi-GPU asynchronous error handling."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TP_SOURCE = (ROOT / "csrc" / "tp_transfer_thread_group.cpp").read_text(
    encoding="utf-8"
)
LAYERWISE_SOURCE = (ROOT / "csrc" / "layerwise.cpp").read_text(encoding="utf-8")


def test_tp_workers_use_per_gpu_error_slots_and_drain_every_future():
    assert "std::vector<std::string> gpu_errors(num_gpus_);" in TP_SOURCE
    assert "std::vector<std::pair<int, std::future<void>>> futures;" in TP_SOURCE
    assert "for (auto &entry : futures)" in TP_SOURCE
    assert "entry.second.get();" in TP_SOURCE
    assert "worker future failed: unknown exception" in TP_SOURCE
    assert "unknown worker exception" in TP_SOURCE

    # These shared writes were a data race when multiple workers failed.
    assert "std::atomic<bool> failed{false};" not in TP_SOURCE
    assert "std::string error_msg;" not in TP_SOURCE


def test_layerwise_workers_share_one_complete_future_drain_helper():
    assert "using GpuFuture = std::pair<int, std::future<void>>;" in (
        LAYERWISE_SOURCE
    )
    assert "void drain_gpu_futures(" in LAYERWISE_SOURCE
    assert "for (auto &entry : futures)" in LAYERWISE_SOURCE
    assert "entry.second.get();" in LAYERWISE_SOURCE

    # One helper definition plus single-group event/submit and multi-group
    # event/submit call sites.
    assert LAYERWISE_SOURCE.count("drain_gpu_futures(") == 5
    assert LAYERWISE_SOURCE.count("throw_if_gpu_errors(") == 5
    assert LAYERWISE_SOURCE.count("unknown worker exception") == 2

    assert "std::atomic<bool> failed{false};" not in LAYERWISE_SOURCE
    assert "std::string error_msg;" not in LAYERWISE_SOURCE


def test_layerwise_poll_event_creation_reports_runtime_failures():
    assert LAYERWISE_SOURCE.count(
        "const cudaError_t status = cudaEventCreateWithFlags("
    ) == 2
    assert LAYERWISE_SOURCE.count(
        '"cudaEventCreateWithFlags failed: "'
    ) == 2
