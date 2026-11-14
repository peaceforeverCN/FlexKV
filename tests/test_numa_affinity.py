"""FlexKV must not be confined to the CPUs of the SGLang rank that launched it.

SGLang's ``bind_dsv41_prefill_worker`` pins every thread of a prefill rank to
4 physical cores. FlexKV's KVServer / TransferManager are spawned from that
rank and inherit the mask, so the FlexKV scheduler thread, the transfer
workers and Mooncake's RDMA threads end up time-sharing those 4 cores with
the SGLang rank itself.

``reset_inherited_numa_binding`` used to return early whenever
``get_spread_nodes()`` was empty, which is exactly what happens on a
single-NUMA host -- so there the fix never ran. These tests reproduce that
chain (parent narrows its affinity, child inherits it and resets) and cover
both topologies on the same hardware: ``FLEXKV_NUMA_NODES=0`` leaves one
usable node, which is what a single-socket host looks like.

Two levels are tested separately, because they promise different things:

* the default widens the mask to every CPU of the usable nodes, so the
  scheduler *may* run FlexKV on cores no rank is using;
* ``FLEXKV_CPUS`` / ``FLEXKV_CPUS_PER_NODE`` (the latter counting backwards
  from each node's tail, which SGLang never hands out) make the mask
  provably disjoint from the ranks. SGLang fills ``FLEXKV_CPUS`` in on its
  own via ``numa_utils.plan_dsv41_reserved_cpus``.

``test_contention_on_the_launcher_cores_costs_throughput`` is the quantitative
half: it measures how much work FlexKV-like workers get done while the SGLang
rank keeps its cores busy, with and without the fix.
"""

import json
import os
import subprocess
import sys
import time
from multiprocessing import Process, Value

import pytest

from flexkv.common import numa

pytestmark = pytest.mark.unit

# What SGLang's bind_dsv41_prefill_worker() hands one prefill rank.
LAUNCHER_PHYSICAL_CORES = 4


# ---------------------------------------------------------------------------
# Host probing
# ---------------------------------------------------------------------------
def host_core_groups(node: int):
    """SMT-sibling groups of ``node`` this test process may actually use."""
    return numa.physical_core_groups(numa.node_cpus(node) & os.sched_getaffinity(0))


def usable_nodes():
    return [n for n in numa._read_status_list("Mems_allowed_list")
            if host_core_groups(n)]


NODES = usable_nodes()
NODE0_CORES = len(host_core_groups(NODES[0])) if NODES else 0

needs_cores = pytest.mark.skipif(
    NODE0_CORES < 4 * LAUNCHER_PHYSICAL_CORES,
    reason=f"needs >= {4 * LAUNCHER_PHYSICAL_CORES} usable physical cores on a node")
needs_two_nodes = pytest.mark.skipif(len(NODES) < 2, reason="needs >= 2 NUMA nodes")


def launcher_mask(node: int, cores: int = LAUNCHER_PHYSICAL_CORES):
    """The mask ``bind_dsv41_prefill_worker`` leaves on GPU0's scheduler: the
    first ``cores`` physical cores of the GPU's NUMA node, SMT siblings kept
    together."""
    return set().union(set(), *host_core_groups(node)[:cores])


def node_wide(nodes):
    cpus = set()
    for node in nodes:
        cpus |= numa.node_cpus(node)
    return cpus & os.sched_getaffinity(0)


# ---------------------------------------------------------------------------
# The real chain: launcher pins itself -> FlexKV child inherits -> child resets
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Stands in for the SGLang rank: bind every thread to a few cores (what
# bind_dsv41_prefill_worker does), then exec the FlexKV child -- the same
# execve KVServer.create_server() uses, which carries the mask over.
_LAUNCHER = (
    "import json, os, sys\n"
    "os.sched_setaffinity(0, set(json.loads(sys.argv[1])))\n"
    "os.execv(sys.executable, [sys.executable, '-c', sys.argv[2]])\n"
)

_CHILD = (
    "import json, os, sys\n"
    "from flexkv.common import numa\n"
    # Pretend there is nothing to spread over, i.e. be a single-socket host no
    # matter what this machine looks like. The CPU half of the reset must not
    # be gated on this -- that gate *was* the bug.
    "if os.environ.get('TEST_FORCE_NO_SPREAD'):\n"
    "    numa.get_spread_nodes = lambda: ()\n"
    "inherited = sorted(os.sched_getaffinity(0))\n"
    "numa.reset_inherited_numa_binding('test')\n"
    # flexkv_logger also writes to stdout, hence the marker.
    "sys.stdout.write('@RESULT@' + json.dumps({\n"
    "    'inherited': inherited,\n"
    "    'after': sorted(os.sched_getaffinity(0)),\n"
    "    'usable_nodes': list(numa.get_usable_nodes()),\n"
    "    'spread_nodes': list(numa.get_spread_nodes())}))\n"
)


def spawn_flexkv_child(launcher_cpus, **env):
    """Run ``reset_inherited_numa_binding`` in a process started the way
    KVServer / TransferManager are: from a launcher pinned to a few cores.

    A child keeps the narrowed mask out of the test process, and only a child
    can prove that the mask is inherited in the first place.
    """
    child_env = dict(os.environ)
    for var in (numa.ENV_ENABLE_NUMA_SPREAD, numa.ENV_NUMA_NODES,
                numa.ENV_CPUS, numa.ENV_CPUS_PER_NODE, "TEST_FORCE_NO_SPREAD"):
        child_env.pop(var, None)
    child_env.update({k: str(v) for k, v in env.items()})
    child_env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT] + ([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else []))
    out = subprocess.run(
        [sys.executable, "-c", _LAUNCHER, json.dumps(sorted(launcher_cpus)), _CHILD],
        env=child_env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-4000:]
    assert "@RESULT@" in out.stdout, out.stdout[-4000:] + out.stderr[-4000:]
    result = json.loads(out.stdout.split("@RESULT@", 1)[1])
    assert set(result["inherited"]) == set(launcher_cpus), \
        "the child did not inherit the launcher's mask; test setup is broken"
    return result


@needs_cores
def test_disabled_leaves_flexkv_on_the_launcher_cores():
    """The behaviour being fixed, still reachable with the flag off: FlexKV
    stays on the handful of cores the SGLang rank runs on."""
    launcher = launcher_mask(NODES[0])
    res = spawn_flexkv_child(launcher, **{numa.ENV_ENABLE_NUMA_SPREAD: "0"})
    assert set(res["after"]) == launcher


@needs_cores
def test_reset_is_not_gated_on_memory_spreading():
    """Regression lock, independent of this host's topology: with nothing to
    spread memory over the reset must still widen FlexKV's mask beyond the
    launcher's cores. Gating the whole reset on ``get_spread_nodes()`` is what
    made every single-NUMA host keep sharing them."""
    launcher = launcher_mask(NODES[0])
    res = spawn_flexkv_child(launcher, **{numa.ENV_ENABLE_NUMA_SPREAD: "1",
                                          "TEST_FORCE_NO_SPREAD": "1"})
    assert res["spread_nodes"] == []
    assert set(res["after"]) > launcher


@needs_cores
def test_single_node_host_widens_to_the_whole_node():
    """The real single-socket configuration: ``FLEXKV_NUMA_NODES=0`` leaves one
    usable node, so ``get_spread_nodes()`` is legitimately empty. The CPU half
    must still run, and take the same path a two-node host does."""
    node = NODES[0]
    launcher = launcher_mask(node)
    res = spawn_flexkv_child(
        launcher,
        **{numa.ENV_ENABLE_NUMA_SPREAD: "1", numa.ENV_NUMA_NODES: str(node)})

    assert res["usable_nodes"] == [node]
    assert res["spread_nodes"] == [], "memory spreading must stay off on one node"
    assert set(res["after"]) == node_wide([node])
    assert set(res["after"]) > launcher


@needs_cores
@needs_two_nodes
def test_multi_node_host_widens_to_every_node():
    """Same arithmetic as the single-node case, one node more. Every node has
    to stay available: FlexKV serves the pool regions on all of them."""
    launcher = launcher_mask(NODES[0])
    res = spawn_flexkv_child(launcher, **{numa.ENV_ENABLE_NUMA_SPREAD: "1"})
    after = set(res["after"])
    assert res["spread_nodes"] == NODES
    assert after == node_wide(NODES)
    assert after > launcher
    for node in NODES:
        assert after & numa.node_cpus(node)


@needs_cores
@needs_two_nodes
def test_whole_node_launcher_is_widened_not_moved():
    """``numactl --cpunodebind=0`` is a whole-node binding. FlexKV gains the
    other node's CPUs and keeps this one: being banished to the other socket,
    whose pool regions it also has to serve, would be worse."""
    node = NODES[0]
    whole_node = node_wide([node])
    res = spawn_flexkv_child(whole_node, **{numa.ENV_ENABLE_NUMA_SPREAD: "1"})
    assert set(res["after"]) == node_wide(NODES)


@needs_cores
def test_cpus_per_node_reserves_the_tail():
    """L1: SGLang hands out the head of a node, so counting backwards from the
    tail is disjoint from every rank without knowing anything about them."""
    node = NODES[0]
    launcher = launcher_mask(node)
    reserved = 2
    res = spawn_flexkv_child(launcher, **{
        numa.ENV_ENABLE_NUMA_SPREAD: "1",
        numa.ENV_NUMA_NODES: str(node),
        numa.ENV_CPUS_PER_NODE: str(reserved)})
    expected = set().union(*host_core_groups(node)[-reserved:])
    assert set(res["after"]) == expected
    assert not (expected & launcher)


@needs_cores
def test_explicit_cpu_list_wins():
    """L1 / L2: ``FLEXKV_CPUS`` is used verbatim -- this is also how SGLang
    hands over the cores it gave to no rank."""
    node = NODES[0]
    launcher = launcher_mask(node)
    want = sorted(set().union(*host_core_groups(node)[-3:]))
    res = spawn_flexkv_child(launcher, **{
        numa.ENV_ENABLE_NUMA_SPREAD: "1",
        numa.ENV_CPUS: ",".join(str(c) for c in want)})
    assert res["after"] == want
    assert not (set(want) & launcher)


# ---------------------------------------------------------------------------
# Planning logic, without touching the host
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_topology(monkeypatch):
    """Two nodes, 8 CPUs each, SMT siblings adjacent (cpu 2k / 2k+1)."""
    monkeypatch.setattr(numa, "node_cpus", lambda n: set(range(n * 8, n * 8 + 8)))
    monkeypatch.setattr(numa, "_core_key", lambda cpu: (cpu // 8, cpu // 2))
    for var in (numa.ENV_CPUS, numa.ENV_CPUS_PER_NODE):
        monkeypatch.delenv(var, raising=False)


def test_plan_takes_every_cpu_of_the_usable_nodes(fake_topology):
    """The default: no inference about where the ranks are, and the same
    expression for one node as for two."""
    assert numa.plan_affinity_cpus((0, 1)) == set(range(16))
    assert numa.plan_affinity_cpus((0,)) == set(range(8))
    assert numa.plan_affinity_cpus(()) == set()


def test_plan_cpus_per_node_takes_the_tail(fake_topology, monkeypatch):
    monkeypatch.setenv(numa.ENV_CPUS_PER_NODE, "2")
    assert numa.plan_affinity_cpus((0, 1)) == {4, 5, 6, 7, 12, 13, 14, 15}
    assert numa.plan_affinity_cpus((0,)) == {4, 5, 6, 7}


def test_plan_cpus_per_node_clamped_to_the_node(fake_topology, monkeypatch):
    monkeypatch.setenv(numa.ENV_CPUS_PER_NODE, "99")
    assert numa.plan_affinity_cpus((0,)) == set(range(8))


def test_plan_explicit_list_is_verbatim(fake_topology, monkeypatch):
    """Not intersected with the usable nodes: SGLang computes it from the full
    cpuset, and an operator asking for a CPU is asking for that CPU."""
    monkeypatch.setenv(numa.ENV_CPUS, "2-4,9")
    assert numa.plan_affinity_cpus((0, 1)) == {2, 3, 4, 9}
    monkeypatch.setenv(numa.ENV_CPUS, "100-101")
    assert numa.plan_affinity_cpus((0,)) == {100, 101}


def test_plan_explicit_list_beats_cpus_per_node(fake_topology, monkeypatch):
    monkeypatch.setenv(numa.ENV_CPUS, "9")
    monkeypatch.setenv(numa.ENV_CPUS_PER_NODE, "2")
    assert numa.plan_affinity_cpus((0, 1)) == {9}


@pytest.mark.parametrize("value", ["-1", "x"])
def test_plan_rejects_bad_cpus_per_node(fake_topology, monkeypatch, value):
    monkeypatch.setenv(numa.ENV_CPUS_PER_NODE, value)
    with pytest.raises(ValueError, match=numa.ENV_CPUS_PER_NODE):
        numa.plan_affinity_cpus((0,))


def test_cpus_explicitly_configured(fake_topology, monkeypatch):
    assert not numa.cpus_explicitly_configured()
    monkeypatch.setenv(numa.ENV_CPUS_PER_NODE, "0")
    assert not numa.cpus_explicitly_configured()
    monkeypatch.setenv(numa.ENV_CPUS_PER_NODE, "4")
    assert numa.cpus_explicitly_configured()
    monkeypatch.delenv(numa.ENV_CPUS_PER_NODE)
    monkeypatch.setenv(numa.ENV_CPUS, "0-3")
    assert numa.cpus_explicitly_configured()


def test_physical_core_groups_keeps_smt_siblings_together(fake_topology):
    assert numa.physical_core_groups(range(8)) == [
        {0, 1}, {2, 3}, {4, 5}, {6, 7}]


# ---------------------------------------------------------------------------
# The one thing the kernel can do behind our back
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_cpuset(monkeypatch):
    """A cpuset of 5 CPUs; ``sched_setaffinity`` silently intersects with it
    and fails outright when the intersection is empty, as the kernel does."""
    state = {"mask": {0, 1, 2, 3}}

    def fake_setaffinity(_pid, mask):
        allowed = set(mask) & {0, 1, 2, 3, 4}
        if not allowed:
            raise OSError(22, "Invalid argument")
        state["mask"] = allowed

    monkeypatch.setattr(os, "sched_setaffinity", fake_setaffinity)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: state["mask"])
    return state


def test_apply_affinity_keeps_the_inherited_mask_when_clamped(fake_cpuset):
    """A node-wide default that the cpuset shrinks below the inherited mask is
    strictly worse than doing nothing."""
    inherited = {0, 1, 2, 3}
    numa._apply_affinity("t", {4, 5, 6, 7}, inherited, explicit=False)
    assert fake_cpuset["mask"] == inherited


def test_apply_affinity_keeps_the_inherited_mask_on_error(fake_cpuset):
    inherited = {0, 1, 2, 3}
    numa._apply_affinity("t", {64, 65}, inherited, explicit=False)
    assert fake_cpuset["mask"] == inherited


def test_apply_affinity_honours_an_explicit_budget(fake_cpuset):
    """``FLEXKV_CPUS`` / ``FLEXKV_CPUS_PER_NODE`` are narrow on purpose."""
    numa._apply_affinity("t", {4}, {0, 1, 2, 3}, explicit=True)
    assert fake_cpuset["mask"] == {4}


def test_apply_affinity_widens(fake_cpuset):
    numa._apply_affinity("t", {0, 1, 2, 3, 4}, {0, 1, 2, 3}, explicit=False)
    assert fake_cpuset["mask"] == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# What the contention actually costs
# ---------------------------------------------------------------------------
def _spin(counter, deadline, cpus):
    os.sched_setaffinity(0, cpus)
    local = 0
    while time.monotonic() < deadline:
        for _ in range(20000):
            local += 1
    with counter.get_lock():
        counter.value += local


def measure_flexkv_throughput(flexkv_cpus, sglang_cpus, workers, seconds=2.0):
    """Iterations the FlexKV-like workers complete while the SGLang rank keeps
    every one of its cores busy. Returns iterations per second."""
    counter = Value("q", 0)
    deadline = time.monotonic() + seconds
    load = [Process(target=_spin, args=(Value("q", 0), deadline, sglang_cpus))
            for _ in range(len(sglang_cpus))]
    flexkv = [Process(target=_spin, args=(counter, deadline, flexkv_cpus))
              for _ in range(workers)]
    for p in load + flexkv:
        p.start()
    for p in load + flexkv:
        p.join(seconds + 120)
        assert p.exitcode == 0, f"spinner failed: {p.exitcode}"
    return counter.value / seconds


@needs_cores
def test_contention_on_the_launcher_cores_costs_throughput():
    """Quantifies the regression: FlexKV squeezed onto the SGLang rank's cores
    against FlexKV on the cores the fix gives it.

    One spinner per FlexKV runnable thread -- TransferManager alone runs the
    scheduler thread, the GPU control thread, the polling thread and one
    worker process per GPU, so being outnumbered on 4 cores is realistic.

    The threshold is far below the ratio of CPU counts involved
    (hundreds-to-one on a real host) so a loaded box cannot flip it.
    """
    node = NODES[0]
    sglang_cpus = sorted(launcher_mask(node))
    free_cpus = sorted(node_wide([node]) - set(sglang_cpus))
    workers = 2 * len(sglang_cpus)

    squeezed = measure_flexkv_throughput(sglang_cpus, sglang_cpus, workers)
    separated = measure_flexkv_throughput(free_cpus, sglang_cpus, workers)

    print(f"\n{workers} FlexKV workers on the launcher's {len(sglang_cpus)} cpus: "
          f"{squeezed:.3g} it/s"
          f"\n{workers} FlexKV workers on their own {len(free_cpus)} cpus:     "
          f"{separated:.3g} it/s"
          f"\nspeedup: {separated / squeezed:.1f}x")
    assert separated > 1.5 * squeezed
