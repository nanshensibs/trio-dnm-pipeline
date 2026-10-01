"""LLM adapter layer.

The reasoning core (evidence graph, hypothesis pool, experiment planner) is pure
Python and needs no model. The *agents* need a model to read literature, propose
mechanisms, run debates, and predict experimental outcomes. This module isolates
that dependency behind a small protocol so the framework can run in three modes:

* ``ClaudeAgentClient`` — backed by the Claude Agent SDK, with the scientific
  MCP servers (PubMed, Open Targets, ClinicalTrials, Consensus, alphaXiv, ...)
  attached so agents can pull real evidence.
* ``AnthropicClient`` — a thin Messages-API client if the SDK isn't installed.
* ``OfflineStubClient`` — deterministic canned responses so the whole pipeline
  runs end-to-end (and is unit-testable) with no network and no key.

All clients return parsed JSON when a ``schema`` hint is supplied; the concrete
model-backed clients instruct the model to emit a single JSON object.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class LLMRequest:
    system: str
    prompt: str
    schema_hint: Optional[dict[str, Any]] = None
    max_tokens: int = 4096
    tools: tuple[str, ...] = ()      # names of MCP tool groups this agent may use
    temperature: float = 0.7


class LLMClient(Protocol):
    def complete(self, req: LLMRequest) -> str: ...
    def complete_json(self, req: LLMRequest) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# Model-backed clients
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = os.environ.get("LUPUS_COSCIENTIST_MODEL", "claude-fable-5")


class AnthropicClient:
    """Direct Messages-API client (no MCP tool use). Requires ``anthropic``."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "The 'anthropic' package is required for AnthropicClient. "
                "pip install anthropic, or use OfflineStubClient."
            ) from exc
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(self, req: LLMRequest) -> str:  # pragma: no cover - network
        system = req.system
        if req.schema_hint:
            system += "\n\nRespond with a single valid JSON object and nothing else."
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system=system,
            messages=[{"role": "user", "content": req.prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def complete_json(self, req: LLMRequest) -> dict[str, Any]:  # pragma: no cover
        return _extract_json(self.complete(req))


class ClaudeAgentClient:
    """Claude Agent SDK client with scientific MCP servers attached.

    The SDK gives agents real tool access (literature search, GWAS/eQTL,
    clinical trials, ...). MCP server configuration is passed in by the caller so
    this module stays decoupled from any particular server layout.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        mcp_servers: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "claude-agent-sdk is required for ClaudeAgentClient. "
                "pip install claude-agent-sdk, or use OfflineStubClient."
            ) from exc
        self._sdk = __import__("claude_agent_sdk")
        self.model = model
        self.mcp_servers = mcp_servers or {}

    def complete(self, req: LLMRequest) -> str:  # pragma: no cover - network/SDK
        from claude_agent_sdk import ClaudeAgentOptions, query

        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=req.system,
            mcp_servers=self.mcp_servers,
            allowed_tools=list(req.tools),
            max_turns=8,
        )
        chunks: list[str] = []

        async def _run() -> None:
            async for message in query(prompt=req.prompt, options=options):
                text = getattr(message, "result", None) or getattr(message, "text", None)
                if text:
                    chunks.append(text)

        import asyncio

        asyncio.run(_run())
        return "\n".join(chunks)

    def complete_json(self, req: LLMRequest) -> dict[str, Any]:  # pragma: no cover
        req = LLMRequest(
            system=req.system + "\n\nReturn ONLY a single JSON object.",
            prompt=req.prompt,
            schema_hint=req.schema_hint,
            max_tokens=req.max_tokens,
            tools=req.tools,
            temperature=req.temperature,
        )
        return _extract_json(self.complete(req))


# --------------------------------------------------------------------------- #
# Offline deterministic stub
# --------------------------------------------------------------------------- #

class OfflineStubClient:
    """Deterministic, network-free client.

    It never fabricates literature it presents as real; instead it returns clearly
    schematic placeholder structures so the *orchestration* can be exercised and
    tested. Each agent supplies a ``stub`` callable via the request metadata (see
    ``agent_base.ScientificAgent._offline``) that produces its canned payload.
    """

    def __init__(self, registry: Optional[dict[str, Any]] = None) -> None:
        self._registry = registry or {}

    def register(self, key: str, payload: Any) -> None:
        self._registry[key] = payload

    def complete(self, req: LLMRequest) -> str:
        payload = self._registry.get(req.system[:40], {"note": "offline-stub"})
        return json.dumps(payload)

    def complete_json(self, req: LLMRequest) -> dict[str, Any]:
        return self._registry.get(req.system[:40], {"note": "offline-stub"})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from model text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError("No JSON object found in model response")
