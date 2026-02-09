import httpx
import pytest

from intelclaw.core.events import EventBus
from intelclaw.skills.manager import SkillManager
from intelclaw.teams.mailbox import MailboxMessage, TeamMailbox
from intelclaw.web.server import WebServer


class FakeApp:
    def __init__(self, *, event_bus: EventBus, skills: SkillManager, mailbox: TeamMailbox):
        self.event_bus = event_bus
        self.skills = skills
        self.mailbox = mailbox
        self.tools = None
        self.agent = None
        self.memory = None


@pytest.mark.asyncio
async def test_skills_api_enable_disable_install_and_mailbox(tmp_path):
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

    bus = EventBus()
    skills = SkillManager(None, bus, builtin_dir=builtin, user_dir=user, state_path=state)
    await skills.initialize()
    mailbox = TeamMailbox(bus)

    app = FakeApp(event_bus=bus, skills=skills, mailbox=mailbox)
    server = WebServer(app=app)

    transport = httpx.ASGITransport(app=server.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/skills")
        assert r.status_code == 200
        payload = r.json()
        assert payload["count"] == 2
        ids = {s["id"] for s in payload["skills"]}
        assert ids == {"windows", "whatsapp"}

        # Enable dependent skill
        r2 = await client.post("/api/skills/whatsapp/enable")
        assert r2.status_code == 200
        assert r2.json()["success"] is True

        # Disabling dependency should be blocked (409)
        r3 = await client.post("/api/skills/windows/disable")
        assert r3.status_code == 409
        j3 = r3.json()
        assert j3["success"] is False
        assert "blocked_by" in j3 and "whatsapp" in j3["blocked_by"]

        # Install a new skill into user dir
        manifest_yaml = "\n".join(
            [
                "id: demo",
                "name: Demo",
                "enabled_by_default: false",
                "",
            ]
        )
        r4 = await client.post(
            "/api/skills/install",
            json={"manifest_yaml": manifest_yaml, "agent_md": "Hello", "enable": True},
        )
        assert r4.status_code == 200
        j4 = r4.json()
        assert j4["success"] is True
        assert j4["skill_id"] == "demo"

        r5 = await client.get("/api/skills")
        ids2 = {s["id"] for s in r5.json()["skills"]}
        assert "demo" in ids2

        assert (user / "demo" / "skill.yaml").exists()
        assert (user / "demo" / "AGENT.md").exists()

        # Mailbox endpoint returns per-session messages
        await mailbox.post(
            MailboxMessage(
                session_id="s1",
                kind="note",
                title="Test",
                body="Hello",
                from_agent="team_lead",
            )
        )
        r6 = await client.get("/api/mailbox", params={"session_id": "s1", "limit": 200})
        assert r6.status_code == 200
        j6 = r6.json()
        assert j6["session_id"] == "s1"
        assert j6["count"] == 1
        assert j6["messages"][0]["title"] == "Test"

