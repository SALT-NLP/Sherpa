"""Opt-in capture of serialized HTTP JSON at the API transport boundary.

Never records HTTP headers, URLs or credentials. Messages are preserved verbatim.
Context variables isolate concurrent episodes while the teacher client is shared.
"""

import asyncio
import json
import os
import threading
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

ACTIVE_TRACE: ContextVar[dict | None] = ContextVar(
    "tutor_api_request_trace", default=None
)
_LOCK = threading.Lock()
_PRIVATE = {
    "authorization",
    "api_key",
    "headers",
    "extra_headers",
    "base_url",
    "api_base",
}


def public_payload(value):
    if isinstance(value, dict):
        return {
            k: public_payload(v) for k, v in value.items() if k.lower() not in _PRIVATE
        }
    if isinstance(value, list):
        return [public_payload(v) for v in value]
    return value


def _append(path, record):
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


async def _record(context, record):
    await asyncio.to_thread(
        _append,
        Path(context["path"]),
        {
            "at": datetime.now(UTC).isoformat(),
            "episode_key": context["episode_key"],
            "execution_try": context["execution_try"],
            **record,
        },
    )


def attach(sdk_client, role):
    """Install once on this SDK client's HTTP client, not on any global class."""
    transport = getattr(sdk_client, "_client", None)
    if transport is None or not hasattr(transport, "event_hooks"):
        raise TypeError("API request capture requires an httpx-backed SDK client")
    if getattr(transport, "_tutor_request_capture", False):
        return

    async def before(request):
        context = ACTIVE_TRACE.get()
        if context is None or request.method != "POST":
            return
        payload = json.loads(await request.aread())
        request_id = uuid.uuid4().hex
        request.extensions["tutor_capture_id"] = request_id
        await _record(
            context,
            {
                "event": "request",
                "request_id": request_id,
                "role": role,
                "payload": public_payload(payload),
            },
        )

    async def after(response):
        context = ACTIVE_TRACE.get()
        request_id = response.request.extensions.get("tutor_capture_id")
        if context is None or request_id is None:
            return
        # Eval uses non-streaming completions. SDK can still read the cached body.
        body = json.loads(await response.aread()) if response.is_success else {}
        await _record(
            context,
            {
                "event": "response",
                "request_id": request_id,
                "role": role,
                "status_code": response.status_code,
                "payload": public_payload(body),
                "choices": public_payload(body.get("choices", [])),
                "usage": body.get("usage"),
            },
        )

    transport.event_hooks.setdefault("request", []).append(before)
    transport.event_hooks.setdefault("response", []).append(after)
    transport._tutor_request_capture = True


def attach_wrapper(wrapper, role):
    caller = getattr(wrapper, "caller", None)
    client = getattr(caller, "_client", None)
    if client is not None:
        attach(client, role)
