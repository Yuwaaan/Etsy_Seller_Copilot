"""Generate the "AI Insights" health report shown on the dashboard.

The health *score* and the underlying signals (improving/declining/risks)
are always computed deterministically by ``analytics.health`` -- never
hallucinated. When an LLM is configured, it's used only to turn those
grounded signals into a natural-language executive summary and prioritized
recommendations; without one, simple templated text is used instead, so the
panel still works fully offline.
"""

from dataclasses import dataclass

import pandas as pd
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from etsy_seller_copilot.analytics.health import HealthScore, compute_health_score
from etsy_seller_copilot.config.llm import get_chat_model

_SYSTEM_PROMPT = (
    "You are an experienced e-commerce business analyst reviewing an Etsy shop's "
    "recent performance. You are given pre-computed, factual metrics -- do not "
    "invent numbers beyond what's given. Write a short executive summary and a "
    "prioritized list of concrete, actionable recommendations for the seller."
)


class _NarrativeOutput(BaseModel):
    """Structured LLM output layered on top of the deterministic health score."""

    executive_summary: str = Field(description="2-3 sentence overview of shop performance right now.")
    recommendations: list[str] = Field(description="3-5 concrete, prioritized next steps for the seller.")


@dataclass(frozen=True)
class ShopHealthReport:
    score: int
    score_breakdown: tuple[str, ...]
    improving: tuple[str, ...]
    declining: tuple[str, ...]
    risks: tuple[str, ...]
    executive_summary: str
    recommendations: tuple[str, ...]
    ai_generated: bool  # False when this used the deterministic/templated fallback


def generate_shop_health_report(
    orders: pd.DataFrame,
    period_days: int = 30,
    llm: BaseChatModel | None = None,
) -> ShopHealthReport:
    """Build the AI Insights panel content for ``orders``.

    The score, improving/declining/risk lists, and score breakdown are
    always deterministic (see ``analytics.health``). The executive summary
    and recommendations are LLM-written when a model is available (and the
    call succeeds); otherwise they're generated from simple templates over
    the same deterministic signals, so the panel degrades gracefully rather
    than failing.
    """
    health = compute_health_score(orders, period_days=period_days)
    breakdown = tuple(f"{c.label}: {c.points:+.1f} pts -- {c.explanation}" for c in health.components)

    llm = llm if llm is not None else get_chat_model()
    if llm is not None:
        try:
            narrative = _generate_narrative(llm, health)
            return ShopHealthReport(
                score=health.score,
                score_breakdown=breakdown,
                improving=health.improving,
                declining=health.declining,
                risks=health.risks,
                executive_summary=narrative.executive_summary,
                recommendations=tuple(narrative.recommendations),
                ai_generated=True,
            )
        except Exception:
            pass  # Fall through to the deterministic narrative below.

    return ShopHealthReport(
        score=health.score,
        score_breakdown=breakdown,
        improving=health.improving,
        declining=health.declining,
        risks=health.risks,
        executive_summary=_fallback_summary(health),
        recommendations=_fallback_recommendations(health),
        ai_generated=False,
    )


def _generate_narrative(llm: BaseChatModel, health: HealthScore) -> _NarrativeOutput:
    structured_llm = llm.with_structured_output(_NarrativeOutput)
    context = (
        f"Health score: {health.score}/100\n"
        f"Improving: {list(health.improving)}\n"
        f"Declining: {list(health.declining)}\n"
        f"Risks: {list(health.risks)}\n"
    )
    result = structured_llm.invoke([("system", _SYSTEM_PROMPT), ("user", context)])
    assert isinstance(result, _NarrativeOutput)
    return result


def _fallback_summary(health: HealthScore) -> str:
    if health.score >= 70:
        return f"Shop health looks strong at {health.score}/100."
    if health.score >= 40:
        return f"Shop health is middling at {health.score}/100 -- worth a closer look."
    return f"Shop health is weak at {health.score}/100 -- several metrics need attention."


def _fallback_recommendations(health: HealthScore) -> tuple[str, ...]:
    if health.declining or health.risks:
        return tuple(f"Investigate: {item}" for item in (*health.declining, *health.risks))[:5]
    return ("Keep monitoring revenue, orders, and repeat customer rate month over month.",)
