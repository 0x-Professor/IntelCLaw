import json

import pytest

from intelclaw.skills.manager import SkillManager


@pytest.mark.asyncio
async def test_skill_manager_defaults_and_dependency_block(tmp_path):
    builtin = tmp_path / "skills"
    user = tmp_path / "data" / "skills"
    state = tmp_path / "data" / "skills_state.json"

    (builtin / "windows").mkdir(parents=True, exist_ok=True)
    (builtin / "windows" / "skill.yaml").write_text(
        "\n".join(
            [
                "id: windows",
                "name: Windows",
                "enabled_by_default: true",
                "depends_on: []",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (builtin / "whatsapp").mkdir(parents=True, exist_ok=True)
    (builtin / "whatsapp" / "skill.yaml").write_text(
        "\n".join(
            [
                "id: whatsapp",
                "name: WhatsApp",
                "enabled_by_default: false",
                "depends_on: [windows]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mgr = SkillManager(None, None, builtin_dir=builtin, user_dir=user, state_path=state)
    await mgr.initialize()

    skills = {s["id"]: s for s in await mgr.list_skills()}
    assert skills["windows"]["enabled"] is True
    assert skills["whatsapp"]["enabled"] is False

    # State file should be created and include defaults.
    state_data = json.loads(state.read_text(encoding="utf-8"))
    assert state_data["enabled"]["windows"] is True
    assert state_data["enabled"]["whatsapp"] is False

    # Enabling whatsapp enables dependency chain.
    r = await mgr.enable("whatsapp")
    assert r["success"] is True
    assert await mgr.is_enabled("windows") is True
    assert await mgr.is_enabled("whatsapp") is True

    # Disabling a dependency is blocked while dependent is enabled.
    r2 = await mgr.disable("windows")
    assert r2["success"] is False
    assert "blocked_by" in r2
    assert "whatsapp" in r2["blocked_by"]


@pytest.mark.asyncio
async def test_skill_manager_user_overrides_builtin(tmp_path):
    builtin = tmp_path / "skills"
    user = tmp_path / "data" / "skills"
    state = tmp_path / "data" / "skills_state.json"

    (builtin / "demo").mkdir(parents=True, exist_ok=True)
    (builtin / "demo" / "skill.yaml").write_text(
        "\n".join(
            [
                "id: demo",
                "name: Demo Builtin",
                "enabled_by_default: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (user / "demo").mkdir(parents=True, exist_ok=True)
    (user / "demo" / "skill.yaml").write_text(
        "\n".join(
            [
                "id: demo",
                "name: Demo User",
                "enabled_by_default: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mgr = SkillManager(None, None, builtin_dir=builtin, user_dir=user, state_path=state)
    await mgr.initialize()

    skills = {s["id"]: s for s in await mgr.list_skills()}
    assert skills["demo"]["name"] == "Demo User"
    assert skills["demo"]["source_kind"] == "user"

