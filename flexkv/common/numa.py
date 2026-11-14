"""CPU placement of FlexKV's shared CPU KV pool and of the processes using it.

Problem
-------
FlexKV's KVServer is launched by the ``dp_client_id == 0`` caller (e.g. the
SGLang scheduler of GPU0) and spawns the TransferManager, which allocates the
CPU KV pool shared by *every* GPU and spawns the transfer workers. Linux keeps
the task mempolicy and CPU affinity across fork/exec/spawn, so FlexKV inherits
whatever that one SGLang rank was pinned to. Two separate things go wrong:

* *Memory*: when the launcher is NUMA-bound (``SGLANG_AUTO_NUMA_BIND`` wraps
  each scheduler with ``numactl --cpunodebind=N --membind=N``) the whole pool
  lands on GPU0's node, and every RDMA transfer from it goes through that
  node's NICs.
* *CPU*: when the launcher is pinned to a handful of cores (SGLang's
  ``bind_dsv41_prefill_worker`` gives each prefill rank 4 physical cores and
  binds every one of its threads) FlexKV's scheduler thread, transfer workers
  and Mooncake's RDMA threads end up time-sharing those same 4 cores *with
  the SGLang rank itself*, which costs far more throughput than a remote-node
  pool ever would.

The CPU half does not need two NUMA nodes, so it must not be gated on them --
that was the bug on single-socket hosts, where ``get_spread_nodes()`` is empty
and the whole reset used to short-circuit. Single-NUMA and multi-NUMA hosts
now take exactly the same CPU path; only the memory half differs.

What ``FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD=1`` (the default) does
----------------------------------------------------------------
1. ``reset_inherited_numa_binding``: KVServer / TransferManager drop the
   launcher's binding. CPU affinity is widened to *every* CPU of the usable
   nodes (``plan_affinity_cpus``), so FlexKV's threads can be scheduled on
   the cores no rank is using instead of queueing behind the launcher on its
   own few cores. This runs on single-node hosts too -- it needs no NUMA
   syscall and no second node. With two or more nodes the task mempolicy
   also becomes interleave, so incidental host allocations are no longer
   pinned to GPU0.
2. ``plan_cpu_pool_numa_regions`` + ``bind_range`` (two or more nodes): like
   Mooncake's ``allocate_buffer_numa_segments``, the pool is split into one
   contiguous, KV-block- and page-aligned region per NUMA node, each
   ``mbind``-ed to its node.
3. The Mooncake store backend registers one MR per region (instead of one MR
   for the whole pool). Mooncake labels every MR with the NUMA node of its
   first page (``cpu:N``), so transfers of each region pick that node's NICs.

Widening the mask lets the scheduler keep FlexKV off the busy cores, but it
does not *guarantee* it. Hard isolation is explicit, and comes from whoever
knows the rank layout rather than from FlexKV guessing at it:

* ``FLEXKV_CPUS`` (e.g. ``320-359,680-719``) pins FlexKV to a CPU list.
* ``FLEXKV_CPUS_PER_NODE=K`` reserves the *last* K physical cores of every
  usable node. SGLang's binders hand out the *head* cores of a node
  (``set_gpu_proc_affinity`` uses ``gpu_id * cores_per_rank``,
  ``bind_dsv41_prefill_worker`` uses ``sorted(cores)[ordinal * 4:]``), so a
  node's tail is disjoint from every rank, not just from the launcher.
* SGLang fills ``FLEXKV_CPUS`` in automatically when it pins its ranks to a
  few cores each (``numa_utils.plan_dsv41_reserved_cpus`` exports the cores
  it hands to no rank, on every node, computed in the launcher before
  ``numactl`` narrows a rank to its own node). Either variable being set
  already turns that off, so an explicit choice always wins.

``FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD=0`` keeps the legacy behaviour: the pool
follows the launcher's memory binding and FlexKV runs on the launcher's cores.
``FLEXKV_NUMA_NODES`` (e.g. ``0,1`` or ``0-3``) restricts the nodes used;
default is every node in the cpuset that has CPUs.
"""

import contextlib
import ctypes
import functools
import math
import os
import platform
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from flexkv.common.debug import flexkv_logger

ENV_ENABLE_NUMA_SPREAD = "FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD"
ENV_NUMA_NODES = "FLEXKV_NUMA_NODES"
ENV_CPUS = "FLEXKV_CPUS"
ENV_CPUS_PER_NODE = "FLEXKV_CPUS_PER_NODE"

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")

_MPOL_DEFAULT = 0
_MPOL_PREFERRED = 1
_MPOL_INTERLEAVE = 3

# A region boundary may move by up to half a unit; refuse to spread when that
# could unbalance the regions by more than ~1/16.
_MIN_UNITS_PER_REGION = 8

_SYSCALLS: Dict[str, Dict[str, int]] = {
    "x86_64": {"mbind": 237, "set_mempolicy": 238, "move_pages": 279},
    "aarch64": {"mbind": 235, "set_mempolicy": 237, "move_pages": 239},
}
_SYSCALLS["amd64"] = _SYSCALLS["x86_64"]
_SYSCALLS["arm64"] = _SYSCALLS["aarch64"]

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def numa_spread_enabled() -> bool:
    value = os.getenv(ENV_ENABLE_NUMA_SPREAD, "1").strip()
    if value not in ("0", "1"):
        raise ValueError(
            f"{ENV_ENABLE_NUMA_SPREAD} must be 0 or 1, got {value!r}")
    return value == "1"


def _parse_list(spec: str) -> List[int]:
    out: Set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _read_status_list(key: str) -> List[int]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return _parse_list(line.split(":", 1)[1])
    except (OSError, ValueError):
        pass
    return []


def node_cpus(node: int) -> Set[int]:
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            return set(_parse_list(f.read()))
    except (OSError, ValueError):
        return set()


@functools.lru_cache(maxsize=None)
def get_usable_nodes() -> Tuple[int, ...]:
    """NUMA nodes FlexKV may run on; empty only when the feature is off.

    Unlike :func:`get_spread_nodes` this does not require two nodes: a
    single-socket host still has to get FlexKV off the launcher's cores, and
    that needs no NUMA syscall at all.
    """
    if not numa_spread_enabled():
        return ()
    allowed = _read_status_list("Mems_allowed_list")
    spec = os.getenv(ENV_NUMA_NODES, "").strip()
    if spec:
        try:
            requested = _parse_list(spec)
        except ValueError as e:
            raise ValueError(f"invalid {ENV_NUMA_NODES}={spec!r}") from e
        nodes = [n for n in requested if n in allowed]
        dropped = sorted(set(requested) - set(nodes))
        if dropped:
            flexkv_logger.warning(
                f"[NUMA] {ENV_NUMA_NODES}: nodes {dropped} are not in "
                f"Mems_allowed {allowed}; ignored")
    else:
        # Skip memory-only nodes (CXL/HBM expanders) unless asked for.
        nodes = [n for n in allowed if node_cpus(n)]
    return tuple(nodes)


@functools.lru_cache(maxsize=None)
def get_spread_nodes() -> Tuple[int, ...]:
    """NUMA nodes the CPU pool is spread over; empty when spreading is off
    or there is nothing to spread over (fewer than two usable nodes)."""
    nodes = get_usable_nodes()
    if not nodes:
        return ()
    if platform.machine().lower() not in _SYSCALLS:
        flexkv_logger.warning(
            f"[NUMA] unsupported arch {platform.machine()}; spreading disabled")
        return ()
    return nodes if len(nodes) >= 2 else ()


# ---------------------------------------------------------------------------
# Syscall wrappers
# ---------------------------------------------------------------------------
def _syscall(name: str, *args) -> None:
    nr = _SYSCALLS[platform.machine().lower()][name]
    ret = _libc.syscall(ctypes.c_long(nr), *args)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{name}: {os.strerror(err)}")


def _nodemask(nodes: Sequence[int]):
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    words = max(nodes) // bits + 1
    mask = (ctypes.c_ulong * words)()
    for n in nodes:
        mask[n // bits] |= 1 << (n % bits)
    # The kernel reads maxnode - 1 bits.
    return mask, ctypes.c_ulong(words * bits + 1)


def set_task_mempolicy(mode: int, nodes: Sequence[int] = ()) -> None:
    if mode == _MPOL_DEFAULT:
        _syscall("set_mempolicy", ctypes.c_int(mode), None, ctypes.c_ulong(0))
        return
    mask, maxnode = _nodemask(nodes)
    _syscall("set_mempolicy", ctypes.c_int(mode), mask, maxnode)


def bind_range(addr: int, length: int, node: int) -> None:
    """``mbind`` [addr, addr+length) to ``node`` (MPOL_PREFERRED).

    PREFERRED rather than Mooncake's BIND: a node that runs short spills to
    the others instead of OOM-killing / SIGBUS-ing the server. For shmem
    (the default pool) the policy is stored on the shared object, so pages
    faulted later by any process follow it; for hugetlbfs it is per-VMA, so
    the caller must prefault the range in this process.
    """
    mask, maxnode = _nodemask([node])
    _syscall("mbind", ctypes.c_void_p(addr), ctypes.c_ulong(length),
             ctypes.c_int(_MPOL_PREFERRED), mask, maxnode, ctypes.c_uint(0))


def query_page_nodes(addrs: Sequence[int]) -> List[int]:
    """NUMA node of the page holding each address (negative errno when the
    page is not populated), via move_pages(2) without moving anything."""
    n = len(addrs)
    if n == 0:
        return []
    pages = (ctypes.c_void_p * n)(*[a & ~(PAGE_SIZE - 1) for a in addrs])
    status = (ctypes.c_int * n)()
    _syscall("move_pages", ctypes.c_int(0), ctypes.c_ulong(n), pages, None,
             status, ctypes.c_int(0))
    return list(status)


def touch_pages(addrs: Sequence[int]) -> None:
    """Map the page holding each address into this process by reading it.

    A read fault on shmem / hugetlbfs maps the existing page (or allocates it
    under the mapping's policy); nothing is written, so this is safe even
    while other processes are writing the pool.
    """
    for a in addrs:
        ctypes.c_uint8.from_address(a).value  # noqa: B018


# ---------------------------------------------------------------------------
# CPU affinity
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _core_key(cpu: int) -> Tuple[int, int]:
    """``(package, core)`` of a CPU; its SMT siblings share the key.

    Unknown topology sorts last as a core of its own, so an unreadable sysfs
    never silently merges unrelated CPUs into one "core".
    """
    topo = f"/sys/devices/system/cpu/cpu{cpu}/topology"
    try:
        with open(f"{topo}/physical_package_id") as f:
            package = int(f.read())
        with open(f"{topo}/core_id") as f:
            core = int(f.read())
    except (OSError, ValueError):
        return (1 << 30, cpu)
    return (package, core)


def physical_core_groups(cpus: Iterable[int]) -> List[Set[int]]:
    """``cpus`` grouped into SMT siblings, ordered by ``(package, core)``."""
    cores: Dict[Tuple[int, int], Set[int]] = {}
    for cpu in cpus:
        cores.setdefault(_core_key(cpu), set()).add(cpu)
    return [cores[key] for key in sorted(cores)]


def _cpus_per_node() -> int:
    spec = os.getenv(ENV_CPUS_PER_NODE, "").strip()
    if not spec:
        return 0
    try:
        count = int(spec)
    except ValueError as e:
        raise ValueError(f"invalid {ENV_CPUS_PER_NODE}={spec!r}") from e
    if count < 0:
        raise ValueError(f"{ENV_CPUS_PER_NODE} must be >= 0, got {count}")
    return count


def plan_affinity_cpus(nodes: Sequence[int]) -> Set[int]:
    """The CPU mask a FlexKV process should run on.

    Default: every CPU of the usable ``nodes``, which is the widest mask
    FlexKV is entitled to and the same arithmetic for one node as for many.
    It does not try to work out which cores the ranks are on -- the mask only
    has to *allow* the idle ones, and the scheduler picks among them far
    better than a guess made once at startup could. Nothing here depends on
    the rank count, on how many cores each rank took or on where they sit.

    Hard isolation is explicit instead of inferred, because the one component
    that knows the rank layout is the launcher, not FlexKV:

    * ``FLEXKV_CPUS`` (e.g. ``320-359,680-719``) is used verbatim. SGLang
      fills it in by itself when it pins its ranks (see
      ``numa_utils.plan_dsv41_reserved_cpus``), so the zero-config default
      already avoids the rank cores on such a host.
    * ``FLEXKV_CPUS_PER_NODE=K`` reserves the last K physical cores (with
      their SMT siblings) of every usable node. SGLang's binders hand out the
      *first* cores of a node (``set_gpu_proc_affinity`` uses
      ``gpu_id * cores_per_rank``, ``bind_dsv41_prefill_worker`` uses
      ``sorted(cores)[ordinal * 4:]``), so counting backwards from the tail
      is disjoint from every rank without knowing anything about them.
    """
    explicit = os.getenv(ENV_CPUS, "").strip()
    if explicit:
        cpus = set(_parse_list(explicit))
        if not cpus:
            raise ValueError(f"invalid {ENV_CPUS}={explicit!r}")
        return cpus

    per_node = _cpus_per_node()
    node_wide: Set[int] = set()
    reserved: Set[int] = set()
    for node in nodes:
        cpus = node_cpus(node)
        if not cpus:
            continue
        node_wide |= cpus
        if per_node:
            groups = physical_core_groups(cpus)
            if per_node >= len(groups):
                flexkv_logger.warning(
                    f"[NUMA] {ENV_CPUS_PER_NODE}={per_node} covers all "
                    f"{len(groups)} physical cores of node {node}; FlexKV will "
                    "NOT be isolated from the launcher's cores on it")
            reserved |= set().union(set(), *groups[-per_node:])
    if per_node and reserved:
        flexkv_logger.info(
            f"[NUMA] {ENV_CPUS_PER_NODE}={per_node}: reserving the last "
            f"{per_node} physical cores ({len(reserved)} cpus) of "
            f"node{list(nodes)} for FlexKV")
        return reserved
    return node_wide


def cpus_explicitly_configured() -> bool:
    """Whether ``FLEXKV_CPUS`` / ``FLEXKV_CPUS_PER_NODE`` ask for a specific
    budget, in which case a narrow mask is the intent rather than an accident."""
    return bool(os.getenv(ENV_CPUS, "").strip()) or _cpus_per_node() > 0


def _apply_affinity(role: str, target: Set[int], inherited: Set[int],
                    explicit: bool) -> None:
    """``sched_setaffinity(target)``, kept only if it did not make things worse.

    The kernel intersects every mask with the cpuset without telling us, so
    even the node-wide default can come back narrower than what we inherited
    (a cpuset overlapping the usable nodes only around the launcher's cores).
    That is strictly worse than doing nothing, so it is undone. An explicit
    budget is honoured however small it is -- being narrow is its purpose.
    """
    try:
        os.sched_setaffinity(0, target)
    except OSError as e:
        flexkv_logger.warning(
            f"[NUMA][{role}] sched_setaffinity({len(target)} cpus) failed: {e}; "
            "keeping the inherited affinity")
        return
    got = os.sched_getaffinity(0)
    if not explicit and len(got) < len(inherited):
        flexkv_logger.warning(
            f"[NUMA][{role}] the cpuset clamped the requested {len(target)} cpus "
            f"to {len(got)}, below the inherited {len(inherited)}; keeping the "
            f"inherited affinity (set {ENV_CPUS} to choose explicitly)")
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, inherited)


# ---------------------------------------------------------------------------
# Process-level binding
# ---------------------------------------------------------------------------
def reset_inherited_numa_binding(role: str) -> None:
    """Drop the NUMA / CPU binding inherited from the launcher (see module doc).

    Must run before the CPU pool is allocated and before worker processes /
    threads are created: mempolicy and affinity are inherited at creation.

    The CPU part runs on single-node hosts as well, through the same code path
    as a multi-node one. Memory spreading needs two nodes, but escaping the
    launcher's few cores does not, and that is where the throughput goes.
    """
    if not numa_spread_enabled():
        return
    nodes = get_usable_nodes()
    if not nodes:
        return
    inherited = os.sched_getaffinity(0)
    target = plan_affinity_cpus(nodes)
    if target:
        _apply_affinity(role, target, inherited,
                        explicit=cpus_explicitly_configured())
    else:
        flexkv_logger.warning(
            f"[NUMA][{role}] node{list(nodes)} reported no CPUs; keeping the "
            "inherited affinity")

    spread = get_spread_nodes()
    policy = f"inherited (only node{list(nodes)} usable)"
    if spread:
        try:
            set_task_mempolicy(_MPOL_INTERLEAVE, spread)
            policy = f"interleave{list(spread)}"
        except OSError as e:
            flexkv_logger.warning(
                f"[NUMA][{role}] set_mempolicy(interleave) failed: {e}; keeping "
                "the inherited policy (docker may need --cap-add SYS_NICE)")
            policy = "inherited (set_mempolicy failed)"
    after = os.sched_getaffinity(0)
    if os.getenv(ENV_CPUS, "").strip():
        source = f"{ENV_CPUS}"
    elif _cpus_per_node():
        source = f"{ENV_CPUS_PER_NODE}={_cpus_per_node()}"
    else:
        source = f"all cpus of node{list(nodes)}"
    flexkv_logger.info(
        f"[NUMA][{role}] pid={os.getpid()} reset inherited binding: "
        f"mempolicy={policy}, cpus {len(inherited)} -> {len(after)} ({source})"
        f"{'' if after & inherited else ', disjoint from the launcher'}")


# ---------------------------------------------------------------------------
# Pool layout
# ---------------------------------------------------------------------------
def plan_cpu_pool_numa_regions(
    logical_bytes: int,
    block_bytes: int,
    page_align_bytes: int,
    nodes: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int, int]]:
    """Split a pool into one contiguous region per NUMA node.

    Returns ``[(offset, length, node), ...]`` covering ``[0, logical_bytes)``;
    callers extend the last region over any mapping padding. Every boundary
    is a multiple of ``lcm(block_bytes, page_align_bytes)`` so that no KV
    block straddles two regions (Mooncake cannot transfer a block across MRs)
    and every region starts on a page (mbind / HugePage MR alignment).

    Deterministic in its inputs: the allocator (TransferManager) and the
    Mooncake backend (worker process) both call it and must agree.
    Returns ``[]`` when spreading is off or the pool is too small to split.
    """
    if nodes is None:
        nodes = get_spread_nodes()
    n = len(nodes)
    if n < 2 or logical_bytes <= 0:
        return []
    unit = math.lcm(max(1, int(block_bytes)), max(1, int(page_align_bytes)))
    if logical_bytes < unit * _MIN_UNITS_PER_REGION * n:
        flexkv_logger.warning(
            f"[NUMA] cannot split the CPU pool ({logical_bytes} B) over {n} "
            f"nodes: region boundaries must be multiples of lcm(block={block_bytes} B, "
            f"page/mapping alignment={page_align_bytes} B) = {unit} B, which is "
            "too coarse; not spreading")
        return []
    total_units = logical_bytes // unit
    bounds = [0]
    for i in range(1, n):
        bounds.append(((total_units * i + n // 2) // n) * unit)
    bounds.append(logical_bytes)
    return [(bounds[i], bounds[i + 1] - bounds[i], int(nodes[i]))
            for i in range(n)]


def bind_cpu_pool(base_ptr: int,
                  regions: Sequence[Tuple[int, int, int]],
                  mapped_bytes: int,
                  role: str) -> bool:
    """mbind every planned region; the last one also covers the padding up to
    ``mapped_bytes``. Returns False (pool keeps the inherited policy) on
    failure."""
    try:
        for i, (off, length, node) in enumerate(regions):
            if i == len(regions) - 1:
                length = mapped_bytes - off
            bind_range(base_ptr + off, length, node)
    except OSError as e:
        flexkv_logger.warning(
            f"[NUMA][{role}] mbind failed: {e}; CPU pool is NOT spread over "
            "NUMA nodes (docker may need --cap-add SYS_NICE)")
        return False
    return True


def populate_shmem_fd(fd: int,
                      regions: Sequence[Tuple[int, int, int]],
                      total_bytes: int,
                      chunk_bytes: int = 1 << 30) -> float:
    """Allocate every page of a mbind-ed shmem fd up front; returns seconds.

    Without this the pages are faulted lazily inside ``cudaHostRegister`` /
    ``ibv_reg_mr`` of 16+ processes at once, which measured ~3x slower than
    registering already-populated memory (the legacy pool was populated by
    torch's copy into /dev/shm). shmem fallocate honours the shared policy
    set by mbind, so each region still lands on its node; running each
    region's fallocate on that node's CPUs keeps page zeroing node-local.
    """
    t0 = time.monotonic()
    errors: List[BaseException] = []

    def _run(off: int, length: int, node: int) -> None:
        try:
            cpus = node_cpus(node) & os.sched_getaffinity(0)
            if cpus:
                os.sched_setaffinity(0, cpus)  # this thread only
            end = off + length
            # Chunked so both regions make progress under shmem's inode lock.
            while off < end:
                n = min(chunk_bytes, end - off)
                os.posix_fallocate(fd, off, n)
                off += n
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = []
    for i, (off, length, node) in enumerate(regions):
        if i == len(regions) - 1:
            length = total_bytes - off
        th = threading.Thread(target=_run, args=(off, length, node), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return time.monotonic() - t0


def prefault_cpu_pool(base_ptr: int,
                      regions: Sequence[Tuple[int, int, int]],
                      mapped_bytes: int,
                      page_bytes: int,
                      max_threads: int = 16) -> None:
    """Populate every page, each region by threads running on its node (as
    Mooncake does for HugeTLB segments). Required for hugetlbfs, whose mbind
    policy only applies to faults through this process's mapping."""
    n = len(regions)
    threads_per_node = max(1, max_threads // n)
    errors: List[BaseException] = []

    def _worker(start: int, end: int, node: int) -> None:
        try:
            cpus = node_cpus(node) & os.sched_getaffinity(0)
            if cpus:
                os.sched_setaffinity(0, cpus)  # this thread only
            for addr in range(start, end, page_bytes):
                ctypes.memset(addr, 0, 1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = []
    for i, (off, length, node) in enumerate(regions):
        if i == n - 1:
            length = mapped_bytes - off
        pages = (length + page_bytes - 1) // page_bytes
        per_thread = (pages + threads_per_node - 1) // threads_per_node
        for t in range(threads_per_node):
            begin = off + t * per_thread * page_bytes
            end = min(off + length, begin + per_thread * page_bytes)
            if begin < end:
                th = threading.Thread(
                    target=_worker, args=(base_ptr + begin, base_ptr + end, node),
                    daemon=True)
                th.start()
                threads.append(th)
    for th in threads:
        th.join()
    if errors:
        raise errors[0]


def describe_cpu_pool_placement(base_ptr: int,
                                regions: Sequence[Tuple[int, int, int]],
                                samples_per_region: int = 64) -> str:
    """Human-readable 'planned node -> sampled actual nodes' per region."""
    parts = []
    for off, length, node in regions:
        step = max(PAGE_SIZE, length // samples_per_region)
        addrs = [base_ptr + off + k * step for k in range(samples_per_region)
                 if k * step < length]
        # move_pages only sees pages mapped in this process.
        touch_pages(addrs)
        try:
            status = query_page_nodes(addrs)
        except OSError as e:
            return f"<move_pages failed: {e}>"
        hist: Dict[str, int] = {}
        for s in status:
            key = f"N{s}" if s >= 0 else "unfaulted"
            hist[key] = hist.get(key, 0) + 1
        parts.append(
            f"[off={off} len={length} plan=N{node} sampled={hist}]")
    return " ".join(parts)
