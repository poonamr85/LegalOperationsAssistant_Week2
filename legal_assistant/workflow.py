"""Single-request LangGraph orchestration. No checkpointer, memory, or review interrupts yet."""
from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from legal_assistant import bot
from time import perf_counter
from functools import wraps
from legal_assistant.retriever import get_retriever


class WorkflowState(TypedDict, total=False):
    question: str
    organization: str
    policy_version: str
    passages: list[dict]
    selected_passages: list[dict]
    grade: bot.EvidenceGrade | None
    draft: bot.Draft | None
    claim_validation: bot.ClaimValidation | None
    faithfulness_percent: float | None
    action: str
    answer: str
    escalation_reason: str
    error_message: str
    failed_stage: str
    citation_validation: str
    decision_reason: str
    steps: Annotated[list[str], add]
    timings: dict[str, float]


def failure(exc, stage):
    # Never retain raw exceptions, provider payloads, or keys in graph state.
    return {"error_message": bot.pipeline_error_message(exc, stage), "failed_stage": stage,
            "escalation_reason": f"Pipeline unavailable: {type(exc).__name__}"}


def entry_route(state):
    return "escalate" if bot.requests_human(state["question"]) else "retrieve"


def timed_node(name):
    def decorate(function):
        @wraps(function)
        def wrapped(state):
            start = perf_counter()
            result = function(state)
            result["timings"] = {**state.get("timings", {}), name: round(perf_counter() - start, 4)}
            return result
        return wrapped
    return decorate


@timed_node("retrieval")
def retrieve_node(state):
    if not state["organization"] or not state["policy_version"]:
        return {"action": "index_needed",
                "answer": "The knowledge base is not ready. Please ask the application administrator to refresh the search index.",
                "decision_reason": "No active organization and policy edition", "steps": ["retrieve"]}
    try:
        passages = get_retriever().search(state["question"], state["organization"], state["policy_version"])
        return {"passages": passages, "steps": ["retrieve"]}
    except Exception as exc:
        return {**failure(exc, "search"), "steps": ["retrieve"]}


def after_retrieval(state):
    if state.get("action") == "index_needed":
        return END
    return "grade" if state.get("passages") and not state.get("error_message") else "escalate"


@timed_node("grading")
def grade_node(state):
    try:
        grade = bot.grade_evidence(state["question"], state["passages"])
        selected = [p for p in state["passages"] if p["id"] in set(grade.selected_ids)]
        return {"grade": grade, "selected_passages": selected, "steps": ["grade"]}
    except Exception as exc:
        return {**failure(exc, "evidence grading"), "steps": ["grade"]}


def after_grading(state):
    grade = state.get("grade")
    return "generate" if (not state.get("error_message") and grade is not None
                          and bot.grade_passes(grade, state["passages"])) else "escalate"


@timed_node("generation")
def generate_node(state):
    try:
        draft = bot.generate_answer(state["question"], state["selected_passages"], state["grade"])
        return {"draft": draft, "steps": ["generate"]}
    except Exception as exc:
        return {**failure(exc, "answer generation"), "steps": ["generate"]}


def after_generation(state):
    draft = state.get("draft")
    if state.get("error_message") or draft is None or draft.action == "escalate":
        return "escalate"
    return "validate"


@timed_node("citation validation")
def validate_node(state):
    draft = state["draft"]
    if draft.action == "answer" and not bot.citations_valid(draft, state["selected_passages"]):
        return {"action": "escalate", "escalation_reason": "Answer contains missing or invalid citation IDs",
                "citation_validation": "Failed: missing or invalid IDs", "steps": ["validate"]}
    return {"action": draft.action,
            "citation_validation": "Passed: cited IDs match selected evidence" if draft.action == "answer" else "Not applicable to clarification",
            "decision_reason": state["grade"].reason if draft.action == "answer" else "Essential employee details need clarification",
            "steps": ["validate"]}


def after_validation(state):
    return "escalate" if state["action"] == "escalate" else "check_claims"


@timed_node("claim validation")
def check_claims_node(state):
    try:
        assessment = bot.validate_claim_support(state["question"], state["draft"], state["selected_passages"])
        count = len(assessment.claims)
        score = (round(100 * sum(c.verdict == "supported" for c in assessment.claims) / count, 2)
                 if count and bot.claim_assessment_complete(assessment, state["draft"], state["selected_passages"]) else None)
        result = {"claim_validation": assessment, "faithfulness_percent": score, "steps": ["check_claims"]}
        if not bot.claim_validation_passes(assessment, state["draft"], state["selected_passages"]):
            return {**result, "action": "escalate",
                    "escalation_reason": "Claim support validation failed: unsupported, contradicted, incomplete, or human decision required"}
        # Publish only after validation; rejected drafts remain internal.
        return {**result, "answer": state["draft"].answer}
    except Exception as exc:
        return {**failure(exc, "claim validation"), "action": "escalate", "steps": ["check_claims"]}


def after_claims(state):
    return "escalate" if state.get("action") == "escalate" else END


@timed_node("escalation")
def escalate_node(state):
    reason = state.get("escalation_reason")
    if not reason:
        if bot.requests_human(state["question"]):
            reason = "Employee requested the Legal team"
        elif state.get("grade") and state["grade"].human_judgment_required:
            reason = state["grade"].human_judgment_reason
        elif state.get("grade") and state["grade"].policy_applicability == "not_applicable":
            reason = state["grade"].applicability_reason
        elif not state.get("passages"):
            reason = "No retrieved evidence"
        elif state.get("draft") and state["draft"].action == "escalate":
            reason = "Answer model requested Legal review"
        else:
            reason = state["grade"].reason if state.get("grade") else "Evidence unavailable"
    return {"action": "escalate", "answer": bot.LEGAL_CONTACT_MESSAGE,
            "escalation_reason": reason, "decision_reason": reason, "steps": ["escalate"]}


def build_workflow():
    builder = StateGraph(WorkflowState)
    for name, node in (("retrieve", retrieve_node), ("grade", grade_node),
                       ("generate", generate_node), ("validate", validate_node),
                       ("check_claims", check_claims_node), ("escalate", escalate_node)):
        builder.add_node(name, node)
    builder.add_conditional_edges(START, entry_route, {"retrieve": "retrieve", "escalate": "escalate"})
    builder.add_conditional_edges("retrieve", after_retrieval, {"grade": "grade", "escalate": "escalate", END: END})
    builder.add_conditional_edges("grade", after_grading, {"generate": "generate", "escalate": "escalate"})
    builder.add_conditional_edges("generate", after_generation, {"validate": "validate", "escalate": "escalate"})
    builder.add_conditional_edges("validate", after_validation, {"escalate": "escalate", "check_claims": "check_claims"})
    builder.add_conditional_edges("check_claims", after_claims, {"escalate": "escalate", END: END})
    builder.add_edge("escalate", END)
    return builder.compile()


# The graph definition is reusable; each invocation starts with fresh request state.
workflow = build_workflow()


def run_workflow(question, organization, policy_version):
    start = perf_counter()
    result = workflow.invoke({"question": question, "organization": organization,
                            "policy_version": policy_version, "passages": [], "selected_passages": [],
                            "grade": None, "draft": None, "claim_validation": None,
                            "faithfulness_percent": None, "error_message": "",
                            "escalation_reason": "", "failed_stage": "", "citation_validation": "Not performed",
                            "decision_reason": "", "steps": [], "timings": {}}, {"recursion_limit": 12})
    result["timings"]["total"] = round(perf_counter() - start, 4)
    return result
