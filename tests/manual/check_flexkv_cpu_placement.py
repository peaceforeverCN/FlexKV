"""Ad-hoc check: what CPUs does FlexKV actually get on this host?

Reproduces the production chain of start_1p4d_nvfp4.sh on 10.101.2.220:
sglang's bind_dsv41_prefill_worker pins the GPU0 prefill rank to 4 physical
cores at the head of its NUMA node, then FlexKV's KVServer/TransferManager
are exec'd from it and call reset_inherited_numa_binding().

    python tests/manual/check_flexkv_cpu_placement.py
"""

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from flexkv.common import numa  # noqa: E402

FLEXKV_ENV = ("FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD", "FLEXKV_NUMA_NODES",
              "FLEXKV_CPUS_PER_NODE", "FLEXKV_CPUS")

_CHILD = (
    "import os, sys, json\n"
    "sys.path.insert(0, {root!r})\n"
    "from flexkv.common.numa import reset_inherited_numa_binding\n"
    "reset_inherited_numa_binding({role!r})\n"
    "sys.stdout.write('@' + json.dumps(sorted(os.sched_getaffinity(0))))\n"
).format(root=REPO_ROOT, role="TransferManager")

_LAUNCHER = (
    "import json, os, sys\n"
    "os.sched_setaffinity(0, set(json.loads(sys.argv[1])))\n"
    "os.execv(sys.executable, [sys.executable, '-c', sys.argv[2]])\n"
)


def flexkv_cpus(launcher_cpus, **env):
    child_env = dict(os.environ)
    for var in FLEXKV_ENV:
        child_env.pop(var, None)
    child_env.update(env)
    out = subprocess.run(
        [sys.executable, "-c", _LAUNCHER, json.dumps(sorted(launcher_cpus)), _CHILD],
        env=child_env, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(out.stderr[-4000:])
    return json.loads(out.stdout.split("@", 1)[1])


def main():
    nodes = [n for n in numa._read_status_list("Mems_allowed_list")
             if numa.node_cpus(n)]
    node = nodes[0]
    groups = numa.physical_core_groups(numa.node_cpus(node))
    cores_per_rank, ranks_on_node = 4, 4  # bind_dsv41_prefill_worker, 4 GPUs/node
    launcher = set().union(*groups[:cores_per_rank])
    all_ranks = set().union(*groups[:cores_per_rank * ranks_on_node])
    # What sglang's numa_utils.plan_dsv41_reserved_cpus() exports as
    # FLEXKV_CPUS: on every node, the cores past the ones its ranks took.
    reserved = set()
    for n in nodes:
        node_groups = numa.physical_core_groups(numa.node_cpus(n))
        reserved |= set().union(set(), *node_groups[cores_per_rank * ranks_on_node:])

    print(f"node{node}: {len(groups)} physical cores, "
          f"{len(numa.node_cpus(node))} cpus; usable nodes {nodes}")
    print(f"sglang GPU0 rank pinned to  : {sorted(launcher)}")
    print(f"all {ranks_on_node} ranks on node{node} occupy: {len(all_ranks)} cpus "
          f"({sorted(all_ranks)[0]}-{sorted(all_ranks)[-1]})")
    print()
    for label, env in [
        ("legacy (flag=0)", {"FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD": "0"}),
        ("L0 default", {}),
        ("L0 single-NUMA simulated", {"FLEXKV_NUMA_NODES": str(node)}),
        ("L2 sglang FLEXKV_CPUS", {"FLEXKV_CPUS": ",".join(map(str, sorted(reserved)))}),
        ("L1 CPUS_PER_NODE=64", {"FLEXKV_CPUS_PER_NODE": "64"}),
    ]:
        cpus = set(flexkv_cpus(launcher, **env))
        print(f"{label:26s} flexkv cpus={len(cpus):4d}  "
              f"overlap with GPU0 rank={len(cpus & launcher):3d}  "
              f"overlap with all node{node} ranks={len(cpus & all_ranks):3d}")


if __name__ == "__main__":
    main()
