from backend.app import conversation

POLICY_MATCH = {
    "title": "Refund Eligibility and Processing",
    "source": "refund-policy.md",
    "text": "Refunds require approval before submission.",
}


def test_support_question_uses_rag_and_converse(monkeypatch):
    monkeypatch.setattr(conversation, "search_support_docs", lambda query, limit: [POLICY_MATCH])

    state = conversation.run_conversation_turn("What is the refund policy?")

    assert state["route"] == "converse"
    assert state["status"] == "resolved"
    assert state["citations"] == [POLICY_MATCH]


def test_refund_request_stops_at_approval(monkeypatch):
    monkeypatch.setattr(conversation, "search_support_docs", lambda query, limit: [POLICY_MATCH])

    state = conversation.run_conversation_turn("Can I get a refund?")

    assert state["route"] == "request_approval"
    assert state["status"] == "pending_approval"
    assert "No refund" in state["response"]


def test_cancellation_request_stops_at_approval(monkeypatch):
    monkeypatch.setattr(conversation, "search_support_docs", lambda query, limit: [POLICY_MATCH])

    state = conversation.run_conversation_turn("Please cancel my subscription")

    assert state["route"] == "request_approval"
    assert state["status"] == "pending_approval"


def test_visual_request_waits_for_image():
    state = conversation.run_conversation_turn("Can I send a screenshot?")

    assert state["route"] == "request_image"
    assert state["status"] == "awaiting_image"


def test_vision_placeholder_does_not_claim_analysis():
    state = conversation.run_conversation_turn("Image attached", image_attached=True)

    assert state["route"] == "run_vision"
    assert state["status"] == "vision_not_enabled"
    assert "not analyzed" in state["response"]


def test_uploaded_image_runs_vision_then_rag(monkeypatch):
    vision_requests = []

    class FakeVisionClient:
        class chat:
            class completions:
                @staticmethod
                def create(**request):
                    vision_requests.append(request)
                    return type(
                        "Completion",
                        (),
                        {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "The status light is blinking amber."})()})()]},
                    )()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(conversation, "Groq", lambda **kwargs: FakeVisionClient())
    monkeypatch.setattr(
        conversation,
        "search_support_docs",
        lambda query, limit: [
            {
                "title": "Device Setup and Troubleshooting",
                "source": "device-troubleshooting.md",
                "text": "Check the indicator-light guide.",
            }
        ]
        if "blinking amber" in query
        else [],
    )

    state = conversation.run_conversation_turn(
        "The device will not connect",
        image_attached=True,
        image_data="data:image/png;base64,aW1hZ2U=",
    )

    assert state["route"] == "converse"
    assert state["status"] == "resolved"
    assert state["visual_summary"] == "The status light is blinking amber."
    assert state["citations"][0]["title"] == "Device Setup and Troubleshooting"
    image_requests = [
        request
        for request in vision_requests
        if request["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
    assert len(image_requests) == 1
    assert image_requests[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png")


def test_risky_visual_request_still_stops_for_approval(monkeypatch):
    class FakeVisionClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_):
                    return type(
                        "Completion",
                        (),
                        {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "A damaged item is visible."})()})()]},
                    )()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(conversation, "Groq", lambda **kwargs: FakeVisionClient())
    monkeypatch.setattr(conversation, "search_support_docs", lambda query, limit: [POLICY_MATCH])

    state = conversation.run_conversation_turn(
        "Can I get a refund for this order?",
        image_attached=True,
        image_data="data:image/jpeg;base64,aW1hZ2U=",
    )

    assert state["route"] == "request_approval"
    assert state["status"] == "pending_approval"
    assert "No refund" in state["response"]


def test_escalation_includes_full_transcript():
    history = [
        {"role": "user", "content": "My device is overheating"},
        {"role": "assistant", "content": "Please unplug it"},
    ]

    state = conversation.run_conversation_turn("Please transfer me to a human", history)

    assert state["route"] == "escalate"
    assert state["status"] == "escalated"
    assert state["escalation_summary"] == [
        *history,
        {"role": "user", "content": "Please transfer me to a human"},
    ]


def test_empty_retrieval_escalates(monkeypatch):
    monkeypatch.setattr(conversation, "search_support_docs", lambda query, limit: [])

    state = conversation.run_conversation_turn("A question with no matching policy")

    assert state["route"] == "escalate"
    assert state["status"] == "escalated"