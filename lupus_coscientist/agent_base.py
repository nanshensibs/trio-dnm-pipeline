"""Base class shared by all 28 scientific agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .context import ResearchContext
from .llm import LLMClient, LLMRequest


@dataclass
class AgentResult:
    """What an agent returns after acting on the context."""

    agent: str
    ok: bool = True
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    produced_evidence: int = 0
    produced_hypotheses: int = 0
    eliminated: int = 0


class ScientificAgent:
    """A specialized agent.

    Subclasses either:
      * override :meth:`act` to run deterministic logic over the context
        (the evidence-graph, ranking, negative-evidence, planner, debate,
        skeptic, consensus, and meta-reviewer agents do this), or
      * rely on the default :meth:`act`, which calls the LLM with the agent's
        system prompt and hands the parsed JSON to :meth:`integrate`
        (the knowledge-acquisition and hypothesis-generation agents do this).
    """

    #: unique short id, e.g. "literature"
    agent_id: str = "agent"
    #: human-readable name
    name: str = "Scientific Agent"
    #: layer number 1..6
    layer: int = 0
    #: one-line purpose
    purpose: str = ""
    #: MCP tool identifiers this agent is permitted to use
    tools: tuple[str, ...] = ()
    #: full system prompt
    system_prompt: str = ""

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client

    # -- default LLM-backed behavior ---------------------------------------
    def build_prompt(self, ctx: ResearchContext) -> str:
        """Render the user-turn prompt for this agent from the context."""
        return (
            f"Research question: {ctx.question}\n"
            f"Focus: {ctx.focus or '(none specified)'}\n\n"
            "Follow your role exactly. Return a single JSON object."
        )

    def request(self, ctx: ResearchContext) -> LLMRequest:
        return LLMRequest(
            system=self.system_prompt,
            prompt=self.build_prompt(ctx),
            tools=self.tools,
        )

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        """Fold an LLM payload into the context. Override per agent."""
        return AgentResult(agent=self.agent_id, summary="(no-op integrate)", payload=payload)

    def act(self, ctx: ResearchContext) -> AgentResult:
        """Default: query the model and integrate. Deterministic agents override."""
        if self.client is None:
            payload = self._offline(ctx)
        else:
            payload = self.client.complete_json(self.request(ctx))
        result = self.integrate(ctx, payload)
        ctx.record(self.agent_id, self.layer, result.summary)
        return result

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        """Deterministic placeholder payload for network-free runs. Override."""
        return {}

    # convenience
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} L{self.layer} {self.agent_id}>"
