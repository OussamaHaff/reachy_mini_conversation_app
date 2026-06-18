from types import SimpleNamespace

from reachy_mini_conversation_app import elevenlabs_realtime as el_rt


def test_elevenlabs_agent_config_uses_v3_tts_and_turn_model(monkeypatch) -> None:
    """ElevenLabs agent provisioning should request v3 TTS and turn-taking."""
    captured: dict[str, object] = {}

    class FakeAgents:
        def update(self, *, agent_id: str, **agent_config: object) -> None:
            captured["agent_id"] = agent_id
            captured["agent_config"] = agent_config

    class FakeElevenLabs:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs
            self.conversational_ai = SimpleNamespace(agents=FakeAgents())

    monkeypatch.setattr(el_rt, "ElevenLabs", FakeElevenLabs)
    monkeypatch.setattr(el_rt, "_active_tool_specs", lambda deps: [])
    monkeypatch.setattr(el_rt, "_session_instructions", lambda instance_path: "Reachy instructions")
    monkeypatch.setattr(el_rt.config, el_rt.ELEVENLABS_AGENT_ID_ENV, "agent_existing")

    handler = el_rt.ElevenLabsRealtimeHandler.__new__(el_rt.ElevenLabsRealtimeHandler)
    handler.deps = SimpleNamespace()
    handler.instance_path = None
    handler._voice_override = "cedar"

    assert handler._ensure_agent_sync("test-key") == "agent_existing"

    agent_config = captured["agent_config"]
    conversation_config = agent_config["conversation_config"]  # type: ignore[index]
    assert conversation_config["tts"]["model_id"] == "eleven_v3_conversational"
    assert conversation_config["tts"]["expressive_mode"] is True
    assert conversation_config["turn"]["turn_model"] == "turn_v3"
