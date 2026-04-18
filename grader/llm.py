"""LLM provider abstraction for the llm-judge similarity backend.

Supports OpenAI and Anthropic with env-driven configuration. The SDKs are
imported lazily inside each provider so a missing SDK only fails when that
provider is actually selected.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


class LLMConfigError(RuntimeError):
    """Raised when env vars are missing or invalid for the selected provider."""


class LLMResponseError(RuntimeError):
    """Raised when a provider returns malformed JSON or fails validation."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_s: float = 60.0


class LLMProvider(ABC):
    """One method: take a system + user prompt and JSON schema, return a parsed dict."""

    @abstractmethod
    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str = "judge_response",
    ) -> dict[str, Any]:
        """Return a JSON object conforming to `schema`.

        Raises LLMResponseError on malformed output.
        """


# ---------------------------------------------------------------------------
# Env / config helpers
# ---------------------------------------------------------------------------

def load_dotenv_if_available(path: str = ".env") -> None:
    """Load variables from a `.env` file if python-dotenv is installed.

    Silent no-op when the package isn't available — this is an optional
    convenience for local development, not a hard dependency.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    load_dotenv(path, override=False)


def build_config_from_env() -> LLMConfig:
    """Read env vars and construct an LLMConfig.

    Raises LLMConfigError with an actionable message if required vars are missing.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
    if provider not in ("openai", "anthropic"):
        raise LLMConfigError(
            f"LLM_PROVIDER must be 'openai' or 'anthropic', got {provider!r}"
        )

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise LLMConfigError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
    else:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic"
            )
        model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5").strip() or "claude-opus-4-5"

    return LLMConfig(provider=provider, model=model, api_key=key)


def build_provider(config: LLMConfig) -> LLMProvider:
    """Instantiate a concrete provider from a validated LLMConfig."""
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    raise LLMConfigError(f"Unknown provider: {config.provider!r}")


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Uses OpenAI's `response_format=json_schema` strict mode."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise LLMConfigError(
                "openai package is required for LLM_PROVIDER=openai. "
                "Install with: pip install openai"
            ) from e
        self._client = OpenAI(api_key=config.api_key, timeout=config.timeout_s)

    def chat_json(self, system, user, schema, schema_name="judge_response"):
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = resp.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"OpenAI returned non-JSON content: {content[:300]!r}"
            ) from e


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Uses Anthropic's tool_use with forced tool_choice to obtain strict JSON."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise LLMConfigError(
                "anthropic package is required for LLM_PROVIDER=anthropic. "
                "Install with: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic(api_key=config.api_key, timeout=config.timeout_s)

    def chat_json(self, system, user, schema, schema_name="judge_response"):
        tool = {
            "name": schema_name,
            "description": "Return the structured judgment for the requested findings.",
            "input_schema": schema,
        }
        resp = self._client.messages.create(
            model=self.config.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": schema_name},
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == schema_name:
                return dict(block.input)
        raise LLMResponseError(
            f"Anthropic response did not include a tool_use block for {schema_name!r}"
        )


# ---------------------------------------------------------------------------
# Mock provider (for tests)
# ---------------------------------------------------------------------------

Responder = (
    Callable[[str, str, dict[str, Any]], dict[str, Any]]
    | list[dict[str, Any]]
)


class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests.

    `responder` is either:
      - a list of canned dict responses (consumed in order), or
      - a callable(system, user, schema) -> dict.
    """

    def __init__(self, responder: Responder):
        self._responder = responder
        self._calls: list[tuple[str, str]] = []
        self._idx = 0

    @property
    def calls(self) -> list[tuple[str, str]]:
        return list(self._calls)

    def chat_json(self, system, user, schema, schema_name="judge_response"):
        self._calls.append((system, user))
        if callable(self._responder):
            return self._responder(system, user, schema)
        if self._idx >= len(self._responder):
            raise LLMResponseError("MockLLMProvider ran out of canned responses")
        out = self._responder[self._idx]
        self._idx += 1
        return out
