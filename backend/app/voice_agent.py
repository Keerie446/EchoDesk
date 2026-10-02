import json
import os

import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, inference
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse
from livekit.plugins import groq

load_dotenv()

API_BASE_URL = os.getenv("ECHODESK_API_URL", "http://localhost:8000").rstrip("/")
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "echodesk-agent")


class EchoDeskVoiceAgent(Agent):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            instructions=(
                "You are EchoDesk. Wait for each customer turn to be handled by the "
                "EchoDesk conversation service. Do not answer customer requests directly."
            ),
        )
        self.session_id = session_id
        self.http_client = httpx.AsyncClient(base_url=API_BASE_URL, timeout=45.0)

    async def on_exit(self) -> None:
        await self.http_client.aclose()

    async def on_enter(self) -> None:
        await self.session.say("Hi, you’re speaking with EchoDesk. How can I help?")

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        message = new_message.text_content.strip()
        if not message:
            raise StopResponse()

        history = [
            {"role": item.role, "content": item.text_content}
            for item in turn_ctx.items
            if item.type == "message"
            and item.role in ("user", "assistant")
            and item.text_content.strip()
        ][-60:]
        try:
            response = await self.http_client.post(
                "/conversation/turn",
                json={
                    "session_id": self.session_id,
                    "message": message,
                    "history": history,
                },
            )
            response.raise_for_status()
            reply = response.json()["reply"]
        except (httpx.HTTPError, KeyError, ValueError):
            reply = "I’m having trouble reaching the support service. Please try again, or ask me to connect you to a human."

        await self.session.say(reply)
        raise StopResponse()


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def echodesk_agent(ctx: agents.JobContext) -> None:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key or groq_api_key == "replace-me":
        raise RuntimeError("Set GROQ_API_KEY before starting the EchoDesk voice agent.")

    session_id = ctx.job.metadata or ctx.room.name
    session = AgentSession(
        stt=groq.STT(
            model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
            language="en",
        ),
        llm=groq.LLM(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
        tts=groq.TTS(
            model=os.getenv("GROQ_TTS_MODEL", "playai-tts"),
            voice=os.getenv("GROQ_TTS_VOICE", "Arista-PlayAI"),
        ),
        turn_handling=agents.TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
        allow_interruptions=True,
    )
    voice_agent = EchoDeskVoiceAgent(session_id)
    await session.start(room=ctx.room, agent=voice_agent)

    @ctx.room.local_participant.register_rpc_method("speak_reply")
    async def speak_reply(data: rtc.RpcInvocationData) -> str:
        linked_participant = session.room_io.linked_participant
        if linked_participant is None or data.caller_identity != linked_participant.identity:
            raise PermissionError("Only the active customer can request spoken replies.")

        try:
            payload = json.loads(data.payload)
            reply = str(payload["reply"])[:4000]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("Invalid spoken reply payload.") from error

        updated_context = voice_agent.chat_ctx.copy()
        updated_context.add_message(role="assistant", content=reply)
        await voice_agent.update_chat_ctx(updated_context)
        await session.interrupt()
        session.say(reply)
        return "ok"


if __name__ == "__main__":
    agents.cli.run_app(server)
