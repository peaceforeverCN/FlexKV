import ast
import contextlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import pytest


pytestmark = pytest.mark.unit
SERVER_PATH = Path(__file__).resolve().parents[1] / "flexkv/server/server.py"


@pytest.fixture
def server_api(monkeypatch):
    monkeypatch.delenv("FLEXKV_CUDA_MPS_CLIENT_PRIORITY", raising=False)
    monkeypatch.delenv("CUDA_MPS_CLIENT_PRIORITY", raising=False)
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "KVServer")
    cls.body = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "create_server"]
    namespace = {
        "__file__": str(SERVER_PATH),
        "os": os,
        "mp": Mock(),
        "contextlib": contextlib,
        "textwrap": textwrap,
        "Optional": Optional,
        "ModelConfig": SimpleNamespace,
        "CacheConfig": SimpleNamespace,
        "flexkv_logger": Mock(),
        "KVServerHandle": Mock(side_effect=lambda process: process),
    }
    exec(compile(ast.Module([cls], type_ignores=[]), str(SERVER_PATH), "exec"), namespace)
    namespace["KVServer"]._server_process = Mock()
    return namespace


def _launch(server_api, **kwargs):
    process = Mock(pid=1234)
    with patch("subprocess.Popen", return_value=process) as popen:
        result = server_api["KVServer"].create_server(
            model_config=SimpleNamespace(total_clients=1),
            cache_config=SimpleNamespace(),
            gpu_register_port="/tmp/flexkv-env-test",
            **kwargs,
        )
    return result, popen


@pytest.mark.parametrize("priority", ["0", "1"])
@pytest.mark.parametrize("inherit_env", [False, True])
@pytest.mark.parametrize("parent_priority", [None, "0", "1"])
def test_priority_is_scoped_to_server(
    server_api, monkeypatch, priority, inherit_env, parent_priority,
):
    monkeypatch.setenv("FLEXKV_CUDA_MPS_CLIENT_PRIORITY", priority)
    if parent_priority is not None:
        monkeypatch.setenv("CUDA_MPS_CLIENT_PRIORITY", parent_priority)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/parent-mps")
    monkeypatch.setenv("FLEXKV_TRANSFER_NUM_CTA_H2D", "2")
    previous = dict(os.environ)
    child_env = {"FLEXKV_ENV_TEST": "child"}
    original_child_env = dict(child_env)

    _, popen = _launch(server_api, inherit_env=inherit_env, child_env=child_env)

    popen.assert_called_once()
    env = popen.call_args.kwargs["env"]
    assert env["CUDA_MPS_CLIENT_PRIORITY"] == priority
    assert env["FLEXKV_CUDA_MPS_CLIENT_PRIORITY"] == priority
    assert env["FLEXKV_TRANSFER_NUM_CTA_H2D"] == "2"
    assert env["FLEXKV_ENV_TEST"] == "child"
    assert ("CUDA_MPS_PIPE_DIRECTORY" in env) == inherit_env
    assert dict(os.environ) == previous
    assert child_env == original_child_env
    server_api["flexkv_logger"].info.assert_any_call(
        "KVServer CUDA_MPS_CLIENT_PRIORITY=%s "
        "(from FLEXKV_CUDA_MPS_CLIENT_PRIORITY)",
        priority,
    )


@pytest.mark.parametrize("inherit_env", [False, True])
@pytest.mark.parametrize("parent_priority", [None, "0", "1"])
def test_unset_priority_preserves_environment_rules(
    server_api, monkeypatch, inherit_env, parent_priority,
):
    if parent_priority is not None:
        monkeypatch.setenv("CUDA_MPS_CLIENT_PRIORITY", parent_priority)
    previous = dict(os.environ)
    child_env = {"FLEXKV_ENV_TEST": "child"}

    _, popen = _launch(server_api, inherit_env=inherit_env, child_env=child_env)

    expected = parent_priority if inherit_env else None
    assert popen.call_args.kwargs["env"].get("CUDA_MPS_CLIENT_PRIORITY") == expected
    assert dict(os.environ) == previous
    assert child_env == {"FLEXKV_ENV_TEST": "child"}


def test_default_launch_without_priority_keeps_multiprocessing(server_api):
    _, popen = _launch(server_api)
    popen.assert_not_called()
    server_api["mp"].Process.assert_called_once()
    server_api["mp"].Process.return_value.start.assert_called_once()


def test_priority_selects_subprocess_for_default_launch(server_api, monkeypatch):
    monkeypatch.setenv("FLEXKV_CUDA_MPS_CLIENT_PRIORITY", "1")
    _, popen = _launch(server_api)
    assert popen.call_args.kwargs["env"]["CUDA_MPS_CLIENT_PRIORITY"] == "1"
    server_api["mp"].Process.assert_not_called()


def test_child_priority_overrides_parent_alias_without_mutating_input(
    server_api, monkeypatch,
):
    monkeypatch.setenv("FLEXKV_CUDA_MPS_CLIENT_PRIORITY", "0")
    child_env = {"FLEXKV_CUDA_MPS_CLIENT_PRIORITY": "1", "CUDA_MPS_CLIENT_PRIORITY": "0"}
    previous = dict(child_env)

    _, popen = _launch(server_api, inherit_env=False, child_env=child_env)

    assert popen.call_args.kwargs["env"]["CUDA_MPS_CLIENT_PRIORITY"] == "1"
    assert child_env == previous
    assert os.environ["FLEXKV_CUDA_MPS_CLIENT_PRIORITY"] == "0"
    assert "CUDA_MPS_CLIENT_PRIORITY" not in os.environ


@pytest.mark.parametrize("priority", ["", "2", "-1", "normal", " 1", "1.0"])
def test_invalid_priority_fails_before_spawning(server_api, monkeypatch, priority):
    monkeypatch.setenv("FLEXKV_CUDA_MPS_CLIENT_PRIORITY", priority)
    previous = dict(os.environ)
    with patch("subprocess.Popen") as popen:
        with pytest.raises(ValueError, match="FLEXKV_CUDA_MPS_CLIENT_PRIORITY must be"):
            server_api["KVServer"].create_server(
                model_config=SimpleNamespace(total_clients=1),
                cache_config=SimpleNamespace(),
                gpu_register_port="/tmp/flexkv-env-test",
                inherit_env=False,
            )
    popen.assert_not_called()
    server_api["mp"].set_start_method.assert_not_called()
    server_api["mp"].Process.assert_not_called()
    assert dict(os.environ) == previous


def test_invalid_child_priority_fails_before_spawning(server_api):
    with pytest.raises(ValueError, match="FLEXKV_CUDA_MPS_CLIENT_PRIORITY must be"):
        _launch(server_api, inherit_env=False,
                child_env={"FLEXKV_CUDA_MPS_CLIENT_PRIORITY": "2"})
    server_api["mp"].Process.assert_not_called()


def test_server_environment_reaches_spawned_worker(server_api, monkeypatch):
    monkeypatch.setenv("FLEXKV_CUDA_MPS_CLIENT_PRIORITY", "1")
    monkeypatch.setenv("CUDA_MPS_CLIENT_PRIORITY", "0")
    _, popen = _launch(server_api, inherit_env=False)
    script = textwrap.dedent("""
        import multiprocessing as mp
        import os
        import subprocess
        import sys

        print("server=" + os.environ["CUDA_MPS_CLIENT_PRIORITY"], flush=True)
        worker = mp.get_context("spawn").Process(
            target=subprocess.run,
            args=([sys.executable, "-c",
                   'import os; print("worker=" + os.environ["CUDA_MPS_CLIENT_PRIORITY"])'],),
            kwargs={"check": True},
        )
        worker.start()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            raise RuntimeError("worker did not exit")
        if worker.exitcode != 0:
            raise RuntimeError("worker failed")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popen.call_args.kwargs["env"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.splitlines() == ["server=1", "worker=1"]
    assert os.environ["CUDA_MPS_CLIENT_PRIORITY"] == "0"
