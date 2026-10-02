import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from backend.app import main


def make_upload(data: bytes, content_type: str = "image/png") -> UploadFile:
    return UploadFile(
        file=BytesIO(data),
        filename="shared-image.png",
        headers=Headers({"content-type": content_type}),
    )


def test_vision_endpoint_validates_upload_and_calls_graph(monkeypatch):
    captured = {}

    async def run_in_threadpool(function, *args):
        captured["function"] = function
        captured["args"] = args
        return {
            "response": "Restart the device and retry pairing.",
            "route": "converse",
            "status": "resolved",
            "response_mode": "policy_fallback",
            "visual_summary": "A blinking amber status light is visible.",
            "citations": [],
        }

    monkeypatch.setattr(main, "run_in_threadpool", run_in_threadpool)
    result = asyncio.run(
        main.analyze_uploaded_image(
            image=make_upload(b"\x89PNG\r\n\x1a\nimage-bytes"),
            session_id="voice-session",
            message="The device will not pair",
            history="[]",
        )
    )

    assert captured["args"][0:3] == ("The device will not pair", [], True)
    assert captured["args"][3].startswith("data:image/png;base64,")
    assert result["visual_summary"] == "A blinking amber status light is visible."
    assert result["status"] == "resolved"


def test_vision_endpoint_rejects_mismatched_image_content():
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.analyze_uploaded_image(
                image=make_upload(b"not-a-png"),
                session_id="voice-session",
                message="Check this image",
                history="[]",
            )
        )

    assert error.value.status_code == 415


def test_vision_endpoint_rejects_oversized_image():
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.analyze_uploaded_image(
                image=make_upload(b"\x89PNG\r\n\x1a\n" + b"x" * (main.MAX_IMAGE_BYTES + 1)),
                session_id="voice-session",
                message="Check this image",
                history="[]",
            )
        )

    assert error.value.status_code == 413


def test_vision_endpoint_rejects_invalid_history():
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.analyze_uploaded_image(
                image=make_upload(b"\x89PNG\r\n\x1a\nimage-bytes"),
                session_id="voice-session",
                message="Check this image",
                history="{bad-json}",
            )
        )

    assert error.value.status_code == 400