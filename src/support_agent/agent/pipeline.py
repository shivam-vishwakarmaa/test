"""The actual agent: classify -> retrieve -> draft -> triage.

Each stage is a separate, independently-testable function rather than one
big prompt asking the model to do everything at once, for two reasons argued
in docs/DECISIONS.md #4: (1) classification errors would silently corrupt
retrieval (wrong intent -> irrelevant precedents pulled), so we want to be
able to evaluate classification accuracy on its own; (2) the triage decision
has to be auditable independently of generation quality -- "why did this
escalate" must have an answer that doesn't require re-reading the drafted
reply.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from support_agent.llm.client import LLMClient
from support_agent.retrieval.index import Precedent, PrecedentIndex

INTENT_NAMES = [
    "playback_error", "live_tv_blackout", "content_availability", "billing_payment",
    "account_access", "cancel_refund", "how_to_feature", "app_ui_feedback",
    "ads_experience", "other_non_support",
]
FLAG_NAMES = ["angry", "churn_risk", "repeat_contact", "outage_signal", "needs_account_data"]

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENT_NAMES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "flags": {"type": "array", "items": {"type": "string", "enum": FLAG_NAMES}},
        "rationale": {"type": "string"},
    },
    "required": ["intent", "confidence", "flags", "rationale"],
    "additionalProperties": False,
}

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "used_precedent_indices": {"type": "array", "items": {"type": "integer"}},
        "cannot_ground": {"type": "boolean"},
    },
    "required": ["reply", "used_precedent_indices", "cannot_ground"],
    "additionalProperties": False,
}


def _classify_system(taxonomy: dict) -> str:
    lines = ["You are the intent classifier for a Hulu customer-support triage agent.",
             "Classify the customer's message into EXACTLY ONE of these intents:\n"]
    for intent in taxonomy["intents"]:
        lines.append(f"- {intent['name']}: {intent['summary'].strip()}")
        for rule in intent.get("edge_rules", []):
            lines.append(f"    RULE: {rule}")
    lines.append(
        "\nAlso set any applicable flags from: angry, churn_risk, repeat_contact, "
        "outage_signal, needs_account_data (see definitions below), and give a "
        "confidence in [0,1] reflecting how unambiguous the intent is -- NOT how "
        "important or urgent the message is."
    )
    for flag in taxonomy["flags"]:
        lines.append(f"- {flag['name']}: {flag['definition']}")
    return "\n".join(lines)


@dataclass
class ClassifyResult:
    intent: str
    confidence: float
    flags: list[str]
    rationale: str


def classify(client: LLMClient, model: str, taxonomy: dict, customer_text: str, tag: str = "") -> ClassifyResult:
    system = _classify_system(taxonomy)
    user = f"Customer message:\n{customer_text!r}"
    out = client.complete_json(system, user, CLASSIFY_SCHEMA, model=model, tag=f"{tag}:classify")
    return ClassifyResult(intent=out["intent"], confidence=out["confidence"],
                           flags=out["flags"], rationale=out["rationale"])


DRAFT_SYSTEM = """You are drafting a customer-support reply for Hulu's Twitter \
support account. You are shown the customer's message and a set of PRECEDENTS: \
past customer messages this brand has actually handled, with the brand's real reply. \

Rules:
1. Ground every factual or procedural claim in a precedent. If you reference a \
policy, a troubleshooting step, or a fact, it must come from a precedent or from \
the customer's own message -- never invent one.
2. If NO precedent is close enough to ground a real answer, set cannot_ground=true \
and write a short reply that asks ONE clarifying question instead of guessing.
3. Match the brand's voice: warm, brief, a little informal, never robotic. Twitter-reply \
length -- 1 to 3 sentences.
4. Never promise a specific fix time, a guaranteed refund amount, or any outcome \
you cannot verify from the precedents.
5. Never ask for a password, full payment card number, or other credential.
6. List which precedents (by their 1-based index as shown) you actually drew on \
in used_precedent_indices. An empty list is fine if cannot_ground is true."""


def _draft_user(customer_text: str, precedents: list[Precedent], intent: str, handling: str) -> str:
    prec_block = "\n".join(
        f"  [{i+1}] customer said: {p.customer_text!r}\n      brand replied: {p.brand_reply!r}"
        for i, p in enumerate(precedents)
    ) or "  (none retrieved)"
    return (
        f"Customer message: {customer_text!r}\n"
        f"Classified intent: {intent} (handling policy: {handling})\n\n"
        f"PRECEDENTS:\n{prec_block}"
    )


@dataclass
class DraftResult:
    reply: str
    used_precedent_indices: list[int]
    cannot_ground: bool
    precedents: list[Precedent] = field(default_factory=list)


def draft(
    client: LLMClient, model: str, customer_text: str, precedents: list[Precedent],
    intent: str, handling: str, tag: str = "",
) -> DraftResult:
    user = _draft_user(customer_text, precedents, intent, handling)
    out = client.complete_json(DRAFT_SYSTEM, user, DRAFT_SCHEMA, model=model, tag=f"{tag}:draft")
    return DraftResult(
        reply=out["reply"], used_precedent_indices=out["used_precedent_indices"],
        cannot_ground=out["cannot_ground"], precedents=precedents,
    )


@dataclass
class TriageDecision:
    escalate: bool
    reasons: list[str]


def triage(
    classify_result: ClassifyResult,
    draft_result: DraftResult,
    handling_by_intent: dict[str, str],
    never_auto_intents: set[str],
    min_confidence: float,
    min_grounding: int,
) -> TriageDecision:
    """Deterministic policy layered on top of the LLM's own outputs -- the
    triage decision is NOT itself an LLM call. This is a deliberate choice
    (docs/DECISIONS.md #6): the disposition that decides whether a human ever
    sees this case must be auditable as a plain rule, not another model call
    whose own confidence we'd then have to trust recursively.
    """
    reasons = []
    handling = handling_by_intent.get(classify_result.intent, "escalate")

    if classify_result.intent in never_auto_intents:
        reasons.append(f"intent '{classify_result.intent}' is on the never-auto list")
    if handling == "escalate":
        reasons.append(f"taxonomy handling policy for '{classify_result.intent}' is 'escalate'")
    if classify_result.confidence < min_confidence:
        reasons.append(f"classifier confidence {classify_result.confidence:.2f} < threshold {min_confidence}")
    if draft_result.cannot_ground:
        reasons.append("drafter reported it could not ground a real answer")
    if len(draft_result.used_precedent_indices) < min_grounding and handling in ("auto_answer", "policy_answer"):
        reasons.append(f"only {len(draft_result.used_precedent_indices)} precedent(s) used, below minimum {min_grounding}")
    if "angry" in classify_result.flags:
        reasons.append("flagged angry")
    if "churn_risk" in classify_result.flags:
        reasons.append("flagged churn_risk")
    if "outage_signal" in classify_result.flags:
        reasons.append("flagged outage_signal (possible mass-impact issue)")

    return TriageDecision(escalate=bool(reasons), reasons=reasons)


@dataclass
class AgentOutput:
    case_id: int | None
    customer_text: str
    classify: ClassifyResult
    draft: DraftResult
    triage: TriageDecision


def run_case(
    client: LLMClient,
    agent_model: str,
    taxonomy: dict,
    index: PrecedentIndex,
    customer_text: str,
    query_vec,
    k: int,
    handling_by_intent: dict[str, str],
    never_auto_intents: set[str],
    min_confidence: float,
    min_grounding: int,
    case_id: int | None = None,
    exclude_case_id: int | None = None,
    tag: str = "",
) -> AgentOutput:
    cls = classify(client, agent_model, taxonomy, customer_text, tag=tag)
    precedents = index.query(query_vec, k=k, exclude_case_id=exclude_case_id)
    handling = handling_by_intent.get(cls.intent, "escalate")
    dr = draft(client, agent_model, customer_text, precedents, cls.intent, handling, tag=tag)
    tri = triage(cls, dr, handling_by_intent, never_auto_intents, min_confidence, min_grounding)
    return AgentOutput(case_id=case_id, customer_text=customer_text, classify=cls, draft=dr, triage=tri)
