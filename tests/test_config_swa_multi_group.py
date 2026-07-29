from __future__ import annotations

import pytest

from flexkv.common.config import (
    UserConfig,
    _normalize_swa_multi_group,
    load_user_config_from_env,
)


# ---------------------------------------------------------------------------
# UserConfig.swa_multi_group -- new int-enum semantics {None, 0, 1, 2}
# ---------------------------------------------------------------------------


def test_swa_multi_group_defaults_to_none(monkeypatch) -> None:
    monkeypatch.delenv("FLEXKV_SWA_MULTI_GROUP", raising=False)

    config = load_user_config_from_env()

    # Field default preserved; only the (public) normalization step maps
    # ``None`` to ``2``.
    assert config.swa_multi_group is None


@pytest.mark.parametrize(
    ("value", "normalized"),
    [
        (None, 2),
        (0, 0),
        (1, 1),
        (2, 2),
    ],
)
def test_normalize_swa_multi_group(value, normalized: int) -> None:
    assert _normalize_swa_multi_group(value) == normalized


@pytest.mark.parametrize("raw_value", ["0", "1", "2"])
def test_swa_multi_group_env_override(
    monkeypatch, raw_value: str
) -> None:
    monkeypatch.setenv("FLEXKV_SWA_MULTI_GROUP", raw_value)

    config = load_user_config_from_env()

    assert config.swa_multi_group == int(raw_value)


@pytest.mark.parametrize(
    "raw_value",
    ["3", "-1", "abc", "true", "false", "True", "False", "", " ", "1.0"],
)
def test_swa_multi_group_env_rejects_invalid(
    monkeypatch, raw_value: str
) -> None:
    monkeypatch.setenv("FLEXKV_SWA_MULTI_GROUP", raw_value)

    with pytest.raises(ValueError, match=r"FLEXKV_SWA_MULTI_GROUP"):
        load_user_config_from_env()


@pytest.mark.parametrize("bool_value", [True, False])
def test_swa_multi_group_rejects_bool_with_migration_hint(bool_value) -> None:
    # bool must be rejected explicitly so YAML `true`/`false` cannot silently
    # slip through int(...) coercion; error message must mention the
    # migration mapping.
    with pytest.raises(ValueError) as exc_info:
        UserConfig(swa_multi_group=bool_value)
    msg = str(exc_info.value)
    assert "int enum" in msg
    assert "true -> 2" in msg
    assert "false -> 1" in msg


@pytest.mark.parametrize("bad_value", [3, -1, 100])
def test_swa_multi_group_rejects_out_of_range_int(bad_value) -> None:
    with pytest.raises(ValueError, match=r"\{0, 1, 2\}"):
        UserConfig(swa_multi_group=bad_value)


@pytest.mark.parametrize("bad_value", ["0", "1", "2", "abc", 1.0, [0], (1,)])
def test_swa_multi_group_rejects_non_int_types(bad_value) -> None:
    with pytest.raises(ValueError):
        UserConfig(swa_multi_group=bad_value)
