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

# EchoDesk

## 💡 Project Overview

EchoDesk is a real-time voice and multimodal customer-support agent MVP for AI Build Challenge 2026, PS-05. Customers can talk to the agent, interrupt it, share a photo or one-frame screen snapshot during the call, and receive policy-grounded guidance with citations.

The LangGraph flow routes support questions through Qdrant retrieval, image uploads through on-demand vision analysis, and unresolved requests to escalation with conversation history. Refund, cancellation, and account-change requests visibly stop at `pending_approval`; the MVP does not execute those actions.

Demo scenarios:

1. Ask about a billing dispute and get the matching policy citation.
2. Describe a device-pairing problem, share a photo/screen snapshot, and get a troubleshooting response on the same call.
3. Request a refund and confirm the response stays pending approval without issuing one.

## 🛠️ Technologies Used

- **Frontend:** Vite, JavaScript, LiveKit JavaScript SDK
- **Backend:** FastAPI, Python
- **Voice transport and interruption handling:** LiveKit WebRTC and LiveKit Agents
- **Speech recognition:** Groq Whisper
- **Conversation model:** Groq-hosted `openai/gpt-oss-20b` for the current account; this account did not expose a conversational Llama model when tested
- **Speech output:** LiveKit Inference Cartesia TTS
- **Vision:** Groq vision-capable model, called only when an image is uploaded
- **Orchestration:** LangGraph
- **Retrieval:** Qdrant with FastEmbed embeddings and seven sample support-policy documents

## ⚙️ Setup & Installation

1. Clone the repository and enter it:

	```sh
	git clone https://github.com/Keerie446/EchoDesk.git
	cd EchoDesk
	```

2. Create a local environment file and add provider credentials:

	```sh
	cp .env.example .env
	```

	Set `GROQ_API_KEY`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` in `.env`. Never commit `.env` or paste secrets into chat. Leave `QDRANT_URL` blank to use local persistent Qdrant; set it and `QDRANT_API_KEY` only when using Qdrant Cloud.

3. Install backend dependencies:

	```sh
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	```

4. Install frontend dependencies:

	```sh
	cd frontend
	npm install
	cd ..
	```

5. For an internet-accessible demo without a paid app host, install Cloudflare Tunnel (`brew install cloudflared`). Vercel serves only the static frontend; FastAPI, local Qdrant, and the LiveKit agent worker still need a running host. See **How to Run** below.

## 🚀 How to Run the Project

### Run locally

Start each service in a separate terminal from the repository root:

```sh
.venv/bin/uvicorn backend.app.main:app --reload --port 8000
```

```sh
.venv/bin/python -m backend.app.voice_agent dev
```

```sh
cd frontend && npm run dev
```

Open the Vite URL printed in the terminal, allow microphone access, and click **Start voice call**. While connected, use **Share image** or **Share screen**. Image uploads accept PNG, JPEG, or WebP up to 4 MB. The first RAG request downloads the FastEmbed `all-MiniLM-L6-v2` model and indexes the support policies under `.qdrant/`.

API health: `http://localhost:8000/health`.

Run automated tests from the repository root:

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

### Deploy the frontend on Vercel (free)

Vercel hosts the Vite frontend; it does **not** run the persistent local-Qdrant API or the long-running LiveKit agent worker in this setup.

1. Start the API and agent worker using the commands above.
2. In another terminal, run `cloudflared tunnel --url http://localhost:8000`. Keep it running and copy the generated HTTPS `trycloudflare.com` URL.
3. Import `Keerie446/EchoDesk` in Vercel and set **Root Directory** to `frontend`. Vercel detects Vite; build command is `npm run build`, output directory is `dist`. `frontend/vercel.json` configures SPA routing and browser microphone/screen permissions.
4. Set Vercel project environment variable `VITE_API_BASE_URL` to the tunnel URL and deploy.
5. Add both `http://localhost:5173` and the Vercel site origin to local `FRONTEND_ORIGINS` in `.env`, then restart FastAPI.
6. Open the Vercel URL and allow microphone access. Your computer, API, worker, and tunnel must stay online for voice and vision features.

Cloudflare Quick Tunnel URLs change and are intended for demos, not stable hosting. The tunnel publicly exposes the local API without user authentication; stop it after the demo. LiveKit media uses the configured LiveKit project. No provider secrets belong in Vercel's frontend environment; only the public API URL is needed there.