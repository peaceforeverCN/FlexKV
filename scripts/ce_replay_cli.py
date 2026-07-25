#!/usr/bin/env python3
"""CLI entry point for CE trace replay.

Usage::

    # Summary of trace file
    python -m scripts.ce_replay_cli /tmp/ce_trace.jsonl --summary

    # Replay entry 0
    python -m scripts.ce_replay_cli /tmp/ce_trace.jsonl --replay 0

    # Replay entry 0 with forced strategy (0=CONTIG_DIRECT, 3=GATHER_SCATTER)
    python -m scripts.ce_replay_cli /tmp/ce_trace.jsonl --replay 0 --force-path 3

    # Replay all entries
    python -m scripts.ce_replay_cli /tmp/ce_trace.jsonl --replay-all

    # Compare all strategies for entry 0
    python -m scripts.ce_replay_cli /tmp/ce_trace.jsonl --compare 0
"""

import argparse
import sys

from flexkv.transfer.ce_replay import CETraceReplayer


PATH_NAMES = [
    "CONTIG_DIRECT", "SEGMENT_DIRECT", "SEGMENT_SCATTER",
    "GATHER_SCATTER", "GATHER_DIRECT",
]


def cmd_summary(replayer: CETraceReplayer) -> None:
    s = replayer.summary()
    print(f"\n{'='*60}")
    print(f"  CE Trace Summary: {replayer.trace_file}")
    print(f"{'='*60}")
    print(f"  Total entries:       {s['total_entries']}")
    print(f"  Directions:          {s.get('directions', {})}")
    print(f"  Strategy distribution: {s.get('strategy_distribution', {})}")
    print(f"  Total transfer:      {s.get('total_transfer_gb', 0):.3f} GB")
    print(f"  Truncated entries:   {s.get('truncated_entries', 0)}")
    print(f"  Layerwise batches:   {s.get('num_batches', 0)}")
    if s.get("time_span_ns", 0) > 0:
        span_s = s["time_span_ns"] / 1e9
        print(f"  Time span:           {span_s:.3f} s")
    print(f"{'='*60}\n")


def cmd_replay(replayer: CETraceReplayer, idx: int,
               force_path: int, path_opt_str: str) -> None:
    path_opt = None
    if path_opt_str == "on":
        path_opt = True
    elif path_opt_str == "off":
        path_opt = False

    r = replayer.replay(
        entry_idx=idx,
        force_path=force_path,
        path_opt=path_opt,
    )
    print(f"\n{'='*60}")
    print(f"  Replay Result (trace_id={r.trace_id})")
    print(f"{'='*60}")
    print(f"  Direction:      {r.direction}")
    batch_id = replayer.entries[idx].batch_id
    if batch_id:
        print(f"  Batch ID:       {batch_id}")
    print(f"  Original path:  {r.original_path}")
    print(f"  Replayed path:  {r.replayed_path}")
    print(f"  Num blocks:     {r.num_blocks}")
    print(f"  Transfer bytes: {r.transfer_bytes} ({r.transfer_bytes/1024:.1f} KB)")
    if r.success:
        print(f"  Elapsed:        {r.elapsed_us:.1f} us")
        print(f"  Bandwidth:      {r.bandwidth_gbs:.2f} GB/s")
    else:
        print(f"  FAILED:         {r.error}")
    print(f"{'='*60}\n")


def cmd_replay_all(replayer: CETraceReplayer, force_path: int,
                   path_opt_str: str) -> None:
    path_opt = None
    if path_opt_str == "on":
        path_opt = True
    elif path_opt_str == "off":
        path_opt = False

    results = replayer.replay_all(force_path=force_path, path_opt=path_opt)

    print(f"\n{'='*80}")
    print(f"  Replay All ({len(results)} entries)")
    print(f"{'='*80}")
    hdr = (f"  {'ID':>5s}  {'Dir':>4s}  {'Orig':>16s}  {'Replay':>16s}  "
           f"{'Time(us)':>10s}  {'BW(GB/s)':>10s}  {'Status':>8s}")
    print(hdr)
    print(f"  {'-'*76}")
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"  {r.trace_id:>5d}  {r.direction:>4s}  {r.original_path:>16s}  "
              f"{r.replayed_path:>16s}  {r.elapsed_us:>10.1f}  "
              f"{r.bandwidth_gbs:>10.2f}  {status:>8s}")
    print(f"{'='*80}\n")

    # Summary stats
    ok = [r for r in results if r.success]
    if ok:
        total_bytes = sum(r.transfer_bytes for r in ok)
        total_us = sum(r.elapsed_us for r in ok)
        avg_bw = (total_bytes / total_us * 1e6 / (1024**3)) if total_us > 0 else 0
        print(f"  Successful: {len(ok)}/{len(results)}")
        print(f"  Total transfer: {total_bytes/1024:.1f} KB")
        print(f"  Total time: {total_us:.1f} us")
        print(f"  Average bandwidth: {avg_bw:.2f} GB/s")


def cmd_compare(replayer: CETraceReplayer, idx: int) -> None:
    results = replayer.compare_strategies(entry_idx=idx)

    print(f"\n{'='*80}")
    print(f"  Strategy Comparison (trace_id={results[0].trace_id}, "
          f"{results[0].direction}, {results[0].num_blocks} blocks)")
    print(f"{'='*80}")
    hdr = (f"  {'Strategy':>18s}  {'Time(us)':>10s}  {'BW(GB/s)':>10s}  "
           f"{'Speedup':>10s}  {'Status':>8s}")
    print(hdr)
    print(f"  {'-'*70}")

    baseline_us = None
    for r in results:
        status = "OK" if r.success else "FAIL"
        speedup = ""
        if r.success and baseline_us is not None and r.elapsed_us > 0:
            speedup = f"{baseline_us / r.elapsed_us:.2f}x"
        elif r.success and baseline_us is None:
            baseline_us = r.elapsed_us
            speedup = "1.00x (base)"
        print(f"  {r.replayed_path:>18s}  {r.elapsed_us:>10.1f}  "
              f"{r.bandwidth_gbs:>10.2f}  {speedup:>10s}  {status:>8s}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="CE transfer trace replay tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trace_file", help="Path to the JSONL trace file")
    parser.add_argument("--summary", action="store_true",
                        help="Print trace summary")
    parser.add_argument("--replay", type=int, default=None,
                        metavar="IDX", help="Replay a single entry by index")
    parser.add_argument("--replay-all", action="store_true",
                        help="Replay all entries")
    parser.add_argument("--compare", type=int, default=None,
                        metavar="IDX", help="Compare all strategies for one entry")
    parser.add_argument("--force-path", type=int, default=-1,
                        metavar="N", help="Force strategy 0-4 (-1 = auto)")
    parser.add_argument("--path-opt", choices=["on", "off", "auto"],
                        default="auto", help="Override path_opt_enabled")

    args = parser.parse_args()

    if not any([args.summary, args.replay is not None,
                args.replay_all, args.compare is not None]):
        args.summary = True

    replayer = CETraceReplayer(args.trace_file)

    if args.summary:
        cmd_summary(replayer)
    if args.replay is not None:
        cmd_replay(replayer, args.replay, args.force_path, args.path_opt)
    if args.replay_all:
        cmd_replay_all(replayer, args.force_path, args.path_opt)
    if args.compare is not None:
        cmd_compare(replayer, args.compare)


if __name__ == "__main__":
    main()
