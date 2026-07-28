"""Static regression tests for multi-group CE argument wiring.

The worker module imports torch/native dependencies at runtime, so these tests
inspect its AST and stay runnable on a source-only checkout.
"""

import ast
from pathlib import Path


WORKER_SOURCE = (
    Path(__file__).resolve().parents[1] / "flexkv" / "transfer" / "worker.py"
)


def _attribute_path(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _method_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(WORKER_SOURCE.read_text(encoding="utf-8"))
    worker_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "tpGPUCPUTransferWorker"
    )
    for node in worker_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name!r} not found")


def test_multi_group_constructor_forwards_copy_pool_tuning():
    method = _method_node("_init_tp_multi_group")
    constructors = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TPTransferThreadGroup"
    ]
    assert len(constructors) == 1

    keywords = {kw.arg: _attribute_path(kw.value) for kw in constructors[0].keywords}
    assert keywords["ce_gather_threads"] == (
        "GLOBAL_CONFIG_FROM_ENV.ce_gather_threads"
    )
    assert keywords["ce_gather_nt"] == "GLOBAL_CONFIG_FROM_ENV.ce_gather_nt"


def test_multi_group_transfer_forwards_mla_d2h_mode():
    method = _method_node("_transfer_impl")
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tp_group_transfer"
        and any(kw.arg == "mla_d2h_mode" for kw in node.keywords)
    ]
    assert len(calls) == 1

    keyword = next(
        kw for kw in calls[0].keywords if kw.arg == "mla_d2h_mode"
    )
    assert _attribute_path(keyword.value) == "self.mla_d2h_mode"


def test_multi_group_transfer_preserves_shared_rank_slice_stride():
    """CE tuning must not replace the per-rank CPU offset/stride argument."""
    method = _method_node("_transfer_impl")
    call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tp_group_transfer"
        and any(kw.arg == "mla_d2h_mode" for kw in node.keywords)
    )

    cpu_tp_stride = call.args[5]
    assert isinstance(cpu_tp_stride, ast.Subscript)
    assert isinstance(cpu_tp_stride.value, ast.Name)
    assert cpu_tp_stride.value.id == "gp"
    assert isinstance(cpu_tp_stride.slice, ast.Constant)
    assert cpu_tp_stride.slice.value == "cpu_tp_stride"

    # The native path uses this flag to select non-MLA shared rank slices:
    # cpu_startoff_inside_chunks = rank * cpu_tp_stride_in_bytes.
    assert _attribute_path(call.args[11]) == "self.is_mla"
