# EchoDesk

EchoDesk is a real-time voice and multimodal customer-support agent MVP for AI Build Challenge 2026, PS-05. It combines LiveKit WebRTC voice, LangGraph routing, Qdrant policy retrieval, and on-demand Groq image analysis.

## Milestone 1: run locally

1. Create a local environment file: `cp .env.example .env`. Leave `QDRANT_URL` blank to use persistent local Qdrant; set it and a real `QDRANT_API_KEY` to use Qdrant Cloud. Fill `GROQ_API_KEY`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` in `.env` for voice and vision. Keep the API secret local; do not commit `.env` or paste credentials into chat.
2. Install the backend dependencies: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
3. Start the API from the repository root: `.venv/bin/uvicorn backend.app.main:app --reload --port 8000`.
4. In another terminal, start the LiveKit worker from the repository root: `.venv/bin/python -m backend.app.voice_agent dev`.
5. In a third terminal, install and start the frontend: `cd frontend && npm install && npm run dev`.
6. Open the Vite URL printed by the frontend command, allow microphone access, then press the microphone button to start a call.

The API health check is available at `http://localhost:8000/health`. Query the policy index with plain text:

```sh
curl -X POST http://localhost:8000/rag/query \
	-H 'Content-Type: application/json' \
	-d '{"query":"How long do I have to return an unused device?"}'
```

The first RAG query downloads the FastEmbed `all-MiniLM-L6-v2` model (about 90 MB) and indexes the seven sample policies. The local Qdrant database is stored under `.qdrant/`; the API returns source titles and excerpts without an accuracy or performance claim.

## Milestone 3: text conversation graph

Post each customer turn to `/conversation/turn`, reusing the returned `session_id` and passing the returned `transcript` as `history` on the next turn:

```sh
curl -X POST http://localhost:8000/conversation/turn \
	-H 'Content-Type: application/json' \
	-d '{"message":"I want a refund for this order"}'
```

The response includes the final graph route, status, citations, and transcript. Refund, cancellation, and account-change requests stop at `pending_approval`; no action is executed. Requests for photos or screenshots return `awaiting_image` until an image is uploaded through the vision endpoint. Escalations return the supplied conversation history plus the current turn. Conversation history is client-supplied and is not persisted by the backend.

Run the text-graph safety tests with `.venv/bin/pip install -r requirements-dev.txt` followed by `.venv/bin/python -m pytest` from the repository root.

## Milestone 4: LiveKit voice

The browser joins a LiveKit Cloud room using a short-lived token from `/voice/token`; the token can join/publish/subscribe only in its session room and explicitly dispatches `echodesk-agent`. The separate worker uses Groq Whisper for speech recognition, Groq-hosted `openai/gpt-oss-20b` for language generation, and LiveKit Inference Cartesia TTS for speech. The current Groq key's model catalog does not include a conversational Llama model, so GPT-OSS is an explicit account-specific fallback; switch `GROQ_MODEL` to an available Llama model when the Groq account enables one. Each finalized transcript turn is sent to the existing `/conversation/turn` LangGraph endpoint; the voice agent speaks that response rather than answering outside the approval and escalation routing. LiveKit's default voice activity and interruption handling allow the customer to barge in while the agent is speaking.

The API, worker, and frontend are three separate local processes. Keep the API running while the worker is active: with local embedded Qdrant, the voice worker calls the API over HTTP so only the API process opens `.qdrant/`. Live voice cannot start without valid LiveKit Cloud credentials and a Groq API key; no credentials are bundled in this repository.

## Vercel frontend with local API

Vercel hosts the Vite static frontend. It does not host the persistent local-Qdrant API or the long-running LiveKit agent worker in this setup. For a free demo without Docker or a paid app host, keep FastAPI and the worker running on your Mac and expose the API with a temporary Cloudflare Quick Tunnel.

1. Install `cloudflared` with Homebrew if needed: `brew install cloudflared`.
2. Start FastAPI and the LiveKit worker using the local commands above.
3. In another terminal, expose the API: `cloudflared tunnel --url http://localhost:8000`. Copy its generated `https://…trycloudflare.com` URL and keep the tunnel running.
4. In Vercel, import `Keerie446/EchoDesk` from GitHub and set the project **Root Directory** to `frontend`. Vercel should detect Vite; build command is `npm run build`, output directory is `dist`.
5. Add Vercel build environment variable `VITE_API_BASE_URL` with the tunnel URL, then deploy.
6. Add the deployed Vercel origin to local `FRONTEND_ORIGINS` in `.env` (for example, `http://localhost:5173,https://your-project.vercel.app`) and restart FastAPI.
7. Open the Vercel URL, allow microphone access, and start the call. The Mac, API, worker, and tunnel must stay awake/connected for voice and image features.

`frontend/vercel.json` includes SPA rewrites and browser microphone/screen-capture permissions. Quick Tunnels use temporary URLs and are for demo use, not stable public hosting. When the tunnel URL changes, update `VITE_API_BASE_URL` in Vercel and redeploy. The tunnel publicly exposes the local API without user authentication, so shut it down after the demo. LiveKit media still uses the configured LiveKit Cloud project; only the agent worker runs locally.

## Milestone 5: image and screen context

While a voice call is active, use **Share image** to choose a PNG, JPEG, or WebP photo/screenshot, or **Share screen** to grant browser permission for a single screen snapshot. Uploads are limited to 4 MB. The API sends that image to the configured `GROQ_VISION_MODEL` only for the upload, then feeds the visual findings into the LangGraph/Qdrant path. The original upload is not written to disk. The browser shows the shared image, visual summary, and policy citations; the connected voice agent speaks the result over LiveKit RPC. Refund, cancellation, and account-change requests still return `pending_approval` and perform no action.

Vision requires a valid Groq API key and a Groq vision-capable model available to the project. The default model is `meta-llama/llama-4-scout-17b-16e-instruct`; change `GROQ_VISION_MODEL` in `.env` if Groq changes model availability.

## Demo scenarios

Run these scenarios with LiveKit and Groq credentials configured:

1. Plain voice resolution: customer asks why a recent bill is higher; agent retrieves the billing policy and explains the charge.
2. Image-triggered resolution: customer describes a device pairing issue, chooses **Share image** or **Share screen**, and the agent inspects the indicator light, cites the device troubleshooting guide, and speaks the next step on the same call.
3. Approval gate: customer requests a refund; the agent cites the refund policy, presents a pending-approval state, and takes no action before approval.

Any later refund, cancellation, or account change must remain visibly pending until approval is granted.