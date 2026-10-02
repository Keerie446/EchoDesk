import os
import re
from typing import Literal, NotRequired, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from backend.app.rag import search_support_docs

Route = Literal[
    "converse",
    "request_image",
    "run_vision",
    "query_rag",
    "request_approval",
    "escalate",
]


class Turn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ConversationState(TypedDict):
    message: str
    history: list[Turn]
    image_attached: bool
    intent: str
    route: Route
    status: str
    response: str
    response_mode: str
    citations: list[dict[str, str]]
    image_data: NotRequired[str]
    visual_summary: NotRequired[str]
    escalation_summary: NotRequired[list[Turn]]


ESCALATION_PHRASES = (
    "speak to a human",
    "talk to a human",
    "speak to an agent",
    "talk to an agent",
    "transfer me",
    "supervisor",
    "injured",
    "smoke",
    "on fire",
    "account takeover",
    "hacked account",
)
VISUAL_PHRASES = ("photo", "picture", "image", "screenshot", "screen share")
RISKY_ACTION = re.compile(
    r"\b(refund|cancel(?:lation)?|change|update|delete|remove)\b.*\b(my|account|subscription|plan|address|email|payment|order)\b"
    r"|\b(my|account|subscription|plan|address|email|payment|order)\b.*\b(refund|cancel(?:lation)?|change|update|delete|remove)\b"
    r"|\b(?:i want|i need|i'd like|can i get|may i get|can you|could you|please|process|issue|submit|initiate|give me)\b.{0,40}\brefund\b",
    re.IGNORECASE,
)


def classify_request(state: ConversationState) -> dict[str, str]:
    message = state["message"].lower()

    if state.get("image_attached", False):
        if RISKY_ACTION.search(message):
            intent = "risky_visual_account_action"
        elif any(phrase in message for phrase in ESCALATION_PHRASES):
            intent = "human_visual_escalation"
        else:
            intent = "visual_context_received"
        return {"intent": intent, "route": "run_vision"}
    if RISKY_ACTION.search(message):
        return {"intent": "risky_account_action", "route": "query_rag"}
    if any(phrase in message for phrase in ESCALATION_PHRASES):
        return {"intent": "human_escalation", "route": "escalate"}
    if any(phrase in message for phrase in VISUAL_PHRASES):
        return {"intent": "visual_context_needed", "route": "request_image"}
    return {"intent": "support_question", "route": "query_rag"}


def route_after_classification(state: ConversationState) -> Route:
    return state["route"]


def request_image_node(_: ConversationState) -> dict[str, str]:
    return {
        "route": "request_image",
        "status": "awaiting_image",
        "response": "Please share a photo or screenshot so I can understand the issue. You can keep describing it while you do.",
    }


def run_vision_node(state: ConversationState) -> dict[str, str]:
    image_data = state.get("image_data")
    if not image_data:
        return {
            "route": "run_vision",
            "status": "vision_not_enabled",
            "response": "Image analysis requires an uploaded image. I have not analyzed visual content.",
        }

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "replace-me":
        raise RuntimeError("Set GROQ_API_KEY to enable image analysis.")

    completion = Groq(api_key=api_key).chat.completions.create(
        model=os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Inspect this customer-support image. Describe only visible evidence "
                            "relevant to the customer's stated issue. Do not infer hidden causes, "
                            "personal identity, or account facts. Keep the description concise. "
                            f"Customer issue: {state['message']}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=250,
    )
    visual_summary = completion.choices[0].message.content
    if not visual_summary:
        raise RuntimeError("The vision model returned no visual findings.")

    query = f"{state['message']}\nVisible image findings: {visual_summary}"
    return {
        "message": query,
        "visual_summary": visual_summary,
        "route": "query_rag",
        "status": "processing",
        "response_mode": "vision",
    }


def route_after_vision(state: ConversationState) -> Literal["query_rag", "stop"]:
    if state.get("status") == "vision_not_enabled":
        return "stop"
    return "query_rag"


def query_rag_node(state: ConversationState) -> dict[str, object]:
    citations = search_support_docs(state["message"], limit=3)
    return {"citations": citations}


def route_after_retrieval(state: ConversationState) -> Route:
    if state["intent"].startswith("risky_"):
        return "request_approval"
    if state["intent"].startswith("human_"):
        return "escalate"
    if not state.get("citations"):
        return "escalate"
    return "converse"


def generate_reply_node(state: ConversationState) -> dict[str, str]:
    citations = state.get("citations", [])
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key and api_key != "replace-me" and citations:
        context = "\n\n".join(
            f"[{item['source']}] {item['text']}" for item in citations
        )
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are EchoDesk, a concise customer support agent. "
                        "Answer only from the supplied policy context. If it does not answer the question, "
                        "say so and recommend escalation. Never claim to perform refunds, cancellations, "
                        "or account changes; those require human approval. Cite the source filename in your answer."
                    ),
                },
                *[
                    {"role": turn["role"], "content": turn["content"]}
                    for turn in state.get("history", [])[-8:]
                ],
                {"role": "user", "content": f"Policy context:\n{context}\n\nCustomer: {state['message']}"},
            ],
            temperature=0.2,
            max_completion_tokens=512,
            reasoning_effort="low",
        )
        response = completion.choices[0].message.content or "I could not form a response from the policy context."
        return {
            "route": "converse",
            "response": response,
            "response_mode": "groq",
            "status": "resolved",
        }

    if citations:
        first = citations[0]
        response = f"{first['text']} (Source: {first['source']})"
    else:
        response = "I could not find a matching support policy, so I recommend escalating this question."
    return {
        "route": "converse",
        "response": response,
        "response_mode": "policy_fallback",
        "status": "resolved",
    }


def request_approval_node(state: ConversationState) -> dict[str, str]:
    citations = state.get("citations", [])
    source = f" The relevant policy is {citations[0]['source']}." if citations else ""
    return {
        "route": "request_approval",
        "status": "pending_approval",
        "response_mode": "policy_gate",
        "response": (
            "I can prepare this request, but it is pending human approval. "
            "No refund, cancellation, or account change has been submitted."
            f"{source}"
        ),
    }


def escalate_node(state: ConversationState) -> dict[str, object]:
    summary = [*state.get("history", []), {"role": "user", "content": state["message"]}]
    return {
        "route": "escalate",
        "status": "escalated",
        "response_mode": "transcript_handoff",
        "response": "I’m escalating this conversation. The support team will receive the complete transcript so you do not need to repeat yourself.",
        "escalation_summary": summary,
    }


def _build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("classify", classify_request)
    graph.add_node("converse", generate_reply_node)
    graph.add_node("request_image", request_image_node)
    graph.add_node("run_vision", run_vision_node)
    graph.add_node("query_rag", query_rag_node)
    graph.add_node("request_approval", request_approval_node)
    graph.add_node("escalate", escalate_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "converse": "converse",
            "request_image": "request_image",
            "run_vision": "run_vision",
            "query_rag": "query_rag",
            "request_approval": "request_approval",
            "escalate": "escalate",
        },
    )
    graph.add_conditional_edges(
        "run_vision",
        route_after_vision,
        {"query_rag": "query_rag", "stop": END},
    )
    graph.add_conditional_edges(
        "query_rag",
        route_after_retrieval,
        {
            "request_approval": "request_approval",
            "converse": "converse",
            "escalate": "escalate",
        },
    )
    for node in ("converse", "request_image", "request_approval", "escalate"):
        graph.add_edge(node, END)
    return graph.compile()


conversation_graph = _build_graph()


def run_conversation_turn(
    message: str,
    history: list[Turn] | None = None,
    image_attached: bool = False,
    image_data: str | None = None,
) -> ConversationState:
    initial_state: ConversationState = {
        "message": message,
        "history": history or [],
        "image_attached": image_attached,
        "intent": "",
        "route": "query_rag",
        "status": "processing",
        "response": "",
        "response_mode": "",
        "citations": [],
    }
    if image_data:
        initial_state["image_data"] = image_data
    return conversation_graph.invoke(initial_state)
