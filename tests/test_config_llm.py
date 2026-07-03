import pytest

from etsy_seller_copilot.config.llm import _detect_provider, get_chat_model


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with a clean slate: no provider env vars set."""
    for var in ("MODEL_PROVIDER", "MODEL_NAME", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestDetectProvider:
    def test_no_keys_configured_returns_none(self) -> None:
        assert _detect_provider() is None

    def test_infers_openai_from_its_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        assert _detect_provider() == "openai"

    def test_infers_anthropic_from_its_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        assert _detect_provider() == "anthropic"

    def test_anthropic_key_takes_priority_when_both_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        assert _detect_provider() == "anthropic"

    def test_explicit_model_provider_overrides_key_inference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("MODEL_PROVIDER", "openai")

        assert _detect_provider() == "openai"

    def test_unrecognized_explicit_provider_falls_back_to_key_inference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_PROVIDER", "not-a-real-provider")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        assert _detect_provider() == "openai"


class TestGetChatModel:
    def test_returns_none_when_unconfigured(self) -> None:
        assert get_chat_model() is None

    def test_returns_a_model_when_a_key_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        model = get_chat_model()

        assert model is not None
        assert hasattr(model, "invoke")
        assert hasattr(model, "bind_tools")
