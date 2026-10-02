import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from livekit import api as livekit_api
from pydantic import BaseModel, Field, StringConstraints
from starlette.concurrency import run_in_threadpool
from typing import Annotated, Literal

from backend.app.conversation import run_conversation_turn
from backend.app.rag import close_support_docs, search_support_docs

load_dotenv()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    close_support_docs()


app = FastAPI(title="EchoDesk API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "echodesk-api"}


class RagQuery(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    limit: int = Field(default=3, ge=1, le=7)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(min_length=1, max_length=4000)]


class ConversationRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()), max_length=64)
    message: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    history: list[ConversationTurn] = Field(default_factory=list, max_length=60)


class VoiceTokenRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


MAX_IMAGE_BYTES = 4 * 1024 * 1024
IMAGE_SIGNATURES = {
    "image/jpeg": lambda data: data.startswith(b"\xff\xd8\xff"),
    "image/png": lambda data: data.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/webp": lambda data: data.startswith(b"RIFF") and data[8:12] == b"WEBP",
}


@app.post("/rag/query")
async def query_support_docs(request: RagQuery) -> dict[str, object]:
    try:
        matches = search_support_docs(request.query, request.limit)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="RAG search is unavailable. Check Qdrant configuration and model access.",
        ) from error
    return {"query": request.query, "matches": matches}


@app.post("/conversation/turn")
def conversation_turn(request: ConversationRequest) -> dict[str, object]:
    try:
        previous_turns = [turn.model_dump() for turn in request.history]
        state = run_conversation_turn(request.message, previous_turns)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="The conversation could not be completed. Check the RAG service and model configuration.",
        ) from error

    transcript = [
        *previous_turns,
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": state["response"]},
    ]
    return {
        "session_id": request.session_id,
        "reply": state["response"],
        "route": state["route"],
        "status": state["status"],
        "response_mode": state["response_mode"],
        "citations": state.get("citations", []),
        "transcript": transcript,
        "escalation_summary": state.get("escalation_summary"),
    }


@app.post("/vision/analyze")
async def analyze_uploaded_image(
    image: UploadFile = File(...),
    session_id: str = Form(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    message: str = Form(..., min_length=1, max_length=1000),
    history: str = Form("[]", max_length=300000),
) -> dict[str, object]:
    content_type = image.content_type or ""
    signature_check = IMAGE_SIGNATURES.get(content_type)
    if signature_check is None:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WebP image.")

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be smaller than 4 MB.")
    if not signature_check(image_bytes):
        raise HTTPException(status_code=415, detail="Image content does not match its declared format.")

    try:
        history_data = json.loads(history)
        if not isinstance(history_data, list) or len(history_data) > 60:
            raise ValueError("Invalid conversation history")
        previous_turns = [ConversationTurn.model_validate(turn).model_dump() for turn in history_data]
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Conversation history must be a list of valid turns.") from error

    image_data = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    try:
        state = await run_in_threadpool(
            run_conversation_turn,
            message,
            previous_turns,
            True,
            image_data,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Image analysis is unavailable. Check GROQ_API_KEY and GROQ_VISION_MODEL configuration.",
        ) from error

    transcript = [
        *previous_turns,
        {"role": "user", "content": message},
        {"role": "assistant", "content": state["response"]},
    ]
    return {
        "session_id": session_id,
        "reply": state["response"],
        "route": state["route"],
        "status": state["status"],
        "response_mode": state["response_mode"],
        "visual_summary": state.get("visual_summary"),
        "citations": state.get("citations", []),
        "transcript": transcript,
    }


@app.post("/voice/token")
async def create_voice_token(request: VoiceTokenRequest) -> dict[str, str]:
    livekit_url = os.getenv("LIVEKIT_URL", "").strip()
    api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
    api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
    if (
        not livekit_url
        or "your-project" in livekit_url
        or not api_key
        or api_key == "replace-me"
        or not api_secret
        or api_secret == "replace-me"
    ):
        raise HTTPException(
            status_code=503,
            detail="Voice is not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET in .env.",
        )

    room_name = f"echodesk-{request.session_id}"
    participant_identity = f"customer-{uuid4().hex[:12]}"
    agent_name = os.getenv("LIVEKIT_AGENT_NAME", "echodesk-agent").strip()
    token = (
        livekit_api.AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name("EchoDesk customer")
        .with_grants(
            livekit_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(
            livekit_api.RoomConfiguration(
                agents=[
                    livekit_api.RoomAgentDispatch(
                        agent_name=agent_name,
                        metadata=request.session_id,
                    )
                ]
            )
        )
        .to_jwt()
    )
    return {
        "server_url": livekit_url,
        "participant_token": token,
        "room_name": room_name,
        "session_id": request.session_id,
    }