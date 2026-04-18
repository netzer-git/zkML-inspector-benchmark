"""Tests for grader.llm module.

Covers env parsing, provider dispatch, and the mock provider used throughout
the LLM-judge test suite. No real API is ever contacted.
"""

from __future__ import annotations

import pytest

from grader.llm import (
    AnthropicProvider,
    LLMConfig,
    LLMConfigError,
    LLMResponseError,
    MockLLMProvider,
    OpenAIProvider,
    build_config_from_env,
    build_provider,
    load_dotenv_if_available,
)


# ---------------------------------------------------------------------------
# build_config_from_env
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all LLM-related env vars before each test so results are deterministic."""
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY", "OPENAI_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


class TestBuildConfigFromEnv:
    def test_default_provider_is_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = build_config_from_env()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.api_key == "sk-test"

    def test_openai_custom_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")
        cfg = build_config_from_env()
        assert cfg.model == "gpt-4-turbo"

    def test_openai_missing_key_raises(self):
        with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
            build_config_from_env()

    def test_anthropic_default_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        cfg = build_config_from_env()
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-opus-4-5"

    def test_anthropic_custom_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        cfg = build_config_from_env()
        assert cfg.model == "claude-sonnet-4-6"

    def test_anthropic_missing_key_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
            build_config_from_env()

    def test_invalid_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with pytest.raises(LLMConfigError, match="LLM_PROVIDER"):
            build_config_from_env()

    def test_provider_value_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = build_config_from_env()
        assert cfg.provider == "openai"

    def test_provider_value_is_whitespace_trimmed(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "  anthropic  ")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        cfg = build_config_from_env()
        assert cfg.provider == "anthropic"


# ---------------------------------------------------------------------------
# build_provider
# ---------------------------------------------------------------------------

class TestBuildProvider:
    def test_unknown_provider_raises(self):
        bad = LLMConfig(provider="unknown", model="x", api_key="y")
        with pytest.raises(LLMConfigError, match="Unknown provider"):
            build_provider(bad)

    def test_openai_without_sdk_raises_config_error(self, monkeypatch):
        """When openai SDK isn't importable, we surface a helpful LLMConfigError."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("openai not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        with pytest.raises(LLMConfigError, match="openai package"):
            build_provider(cfg)

    def test_anthropic_without_sdk_raises_config_error(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("anthropic not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-5", api_key="sk-ant")
        with pytest.raises(LLMConfigError, match="anthropic package"):
            build_provider(cfg)


# ---------------------------------------------------------------------------
# load_dotenv_if_available
# ---------------------------------------------------------------------------

class TestLoadDotenv:
    def test_no_error_when_package_missing(self, monkeypatch):
        """Function is a best-effort helper — never raises."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dotenv":
                raise ImportError("dotenv not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        load_dotenv_if_available(".nonexistent.env")  # must not raise


# ---------------------------------------------------------------------------
# MockLLMProvider
# ---------------------------------------------------------------------------

class TestMockLLMProvider:
    def test_canned_list_consumed_in_order(self):
        canned = [{"judgments": [{"gt_id": "a"}]}, {"judgments": [{"gt_id": "b"}]}]
        p = MockLLMProvider(canned)
        assert p.chat_json("sys", "u1", {})["judgments"][0]["gt_id"] == "a"
        assert p.chat_json("sys", "u2", {})["judgments"][0]["gt_id"] == "b"

    def test_canned_list_exhausted_raises(self):
        p = MockLLMProvider([{"judgments": []}])
        p.chat_json("sys", "u1", {})
        with pytest.raises(LLMResponseError, match="ran out"):
            p.chat_json("sys", "u2", {})

    def test_callable_responder(self):
        def responder(system, user, schema):
            return {"judgments": [{"gt_id": user}]}

        p = MockLLMProvider(responder)
        out = p.chat_json("sys", "candidate-7", {})
        assert out["judgments"][0]["gt_id"] == "candidate-7"

    def test_records_calls(self):
        p = MockLLMProvider([{"judgments": []}, {"judgments": []}])
        p.chat_json("sys1", "user1", {})
        p.chat_json("sys2", "user2", {})
        assert p.calls == [("sys1", "user1"), ("sys2", "user2")]

    def test_calls_property_returns_copy(self):
        p = MockLLMProvider(lambda s, u, sc: {"judgments": []})
        p.chat_json("s", "u", {})
        snapshot = p.calls
        p.chat_json("s", "u2", {})
        assert len(snapshot) == 1  # snapshot was copied, not a live reference
