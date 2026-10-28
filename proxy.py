#!/usr/bin/env python3
"""
AgentRouter Anthropic-compatible proxy for OpenCode.

Key policies:
  • Single upstream request (no hidden retries that confuse client state).
  • Protocol closure: every message_start emitted downstream is guaranteed to
    be followed by content blocks and a final message_stop. Any upstream
    truncation is repaired by synthesizing the missing tail.
  • Message rollover: if upstream sends a new message_start before closing the
    previous one, the proxy injects a message_stop for the old message first.
  • Deferred start: message_start is held until the first content_block_start
    is seen, preventing dangling starts on zero-content responses.
  • Tool blocks are buffered and normalized into strict JSON objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator

import anthropic
import httpx2
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET = os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org").rstrip("/")
HOST = os.environ.get("AGENTROUTER_HOST", "127.0.0.1")
PORT = int(os.environ.get("AGENTROUTER_PORT", "7187"))
KEY_FILE = Path.home() / ".config/opencode/api_keys/AGENT_ROUTER_API_KEY"

CONNECT_TIMEOUT = float(os.environ.get("AGENTROUTER_CONNECT_TIMEOUT", "15"))
READ_TIMEOUT = float(os.environ.get("AGENTROUTER_READ_TIMEOUT", "900"))
WRITE_TIMEOUT = float(os.environ.get("AGENTROUTER_WRITE_TIMEOUT", "30"))
POOL_TIMEOUT = float(os.environ.get("AGENTROUTER_POOL_TIMEOUT", "10"))
MODEL_ID = os.environ.get("AGENTROUTER_MODEL_ID", "gpt-5.6-sol")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("agentrouter-proxy")

_HTTP_CLIENT = httpx2.Client(
    limits=httpx2.Limits(
        max_connections=int(os.environ.get("AGENTROUTER_MAX_CONNECTIONS", "50")),
        max_keepalive_connections=int(os.environ.get("AGENTROUTER_MAX_KEEPALIVE", "10")),
        keepalive_expiry=20.0,
    ),
    timeout=httpx2.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT,
    ),
)

_app_client_lock = threading.Lock()
_anthropic_client: anthropic.Anthropic | None = None


class ProxyConfigurationError(RuntimeError):
    pass


def _read_server_key() -> str | None:
    key = os.environ.get("AGENTROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        if KEY_FILE.exists():
            key = KEY_FILE.read_text(encoding="utf-8").strip()
            if key:
                return key
    except OSError as exc:
        logger.warning("Could not read API key file (%s)", type(exc).__name__)
    return None


def _get_client(request: Request | None = None) -> anthropic.Anthropic:
    global _anthropic_client
    server_key = _read_server_key()
    if not server_key:
        raise ProxyConfigurationError(
            "No upstream API key configured. Set AGENTROUTER_API_KEY or create "
            f"{KEY_FILE}."
        )
    with _app_client_lock:
        if _anthropic_client is None:
            _anthropic_client = anthropic.Anthropic(
                api_key=server_key,
                base_url=TARGET,
                http_client=_HTTP_CLIENT,
                timeout=READ_TIMEOUT,
                max_retries=0,
            )
        return _anthropic_client


# ---------------------------------------------------------------------------
# Request normalization
# ---------------------------------------------------------------------------

_INTERNAL_REQUEST_FIELDS = {"stream"}
_ALLOWED_TOP_LEVEL_FIELDS = {
    "model", "messages", "system", "max_tokens", "metadata", "stop_sequences",
    "stream", "temperature", "thinking", "output_config", "top_k", "top_p",
    "tools", "tool_choice", "betas", "service_tier",
}


def _normalize_request(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("Request JSON body must be an object")
    if not isinstance(body.get("model"), str) or not body["model"].strip():
        raise ValueError("Request field 'model' must be a non-empty string")
    if not isinstance(body.get("messages"), list):
        raise ValueError("Request field 'messages' must be an array")
    if not isinstance(body.get("max_tokens"), int) or body["max_tokens"] < 1:
        raise ValueError("Request field 'max_tokens' must be a positive integer")

    result = {
        k: v for k, v in body.items()
        if k not in _INTERNAL_REQUEST_FIELDS
    }
    allow_unknown = os.environ.get("AGENTROUTER_ALLOW_UNKNOWN_FIELDS", "0") == "1"
    if not allow_unknown:
        result = {k: v for k, v in result.items() if k in _ALLOWED_TOP_LEVEL_FIELDS}
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _repair_message_start(payload: dict[str, Any]) -> None:
    if payload.get("type") != "message_start":
        return
    msg = payload.get("message")
    if not isinstance(msg, dict):
        msg = {}
        payload["message"] = msg
    if not isinstance(msg.get("id"), str) or not msg["id"]:
        msg["id"] = _new_message_id()
    msg.setdefault("type", "message")
    msg.setdefault("role", "assistant")
    if not isinstance(msg.get("content"), list):
        msg["content"] = []
    if not isinstance(msg.get("model"), str) or not msg["model"]:
        msg["model"] = MODEL_ID
    if not isinstance(msg.get("usage"), dict):
        msg["usage"] = {"input_tokens": 0, "output_tokens": 0}
    else:
        msg["usage"].setdefault("input_tokens", 0)
        msg["usage"].setdefault("output_tokens", 0)
    msg.setdefault("stop_reason", None)
    msg.setdefault("stop_sequence", None)


def _last_valid_json_object(text: str, max_candidates: int = 32) -> str | None:
    if not text:
        return None
    opens = [i for i, ch in enumerate(text) if ch == "{"]
    closes = [i for i, ch in enumerate(text) if ch == "}"]
    tried = 0
    for i in range(len(opens) - 1, -1, -1):
        if tried >= max_candidates: break
        tried += 1
        candidate = text[opens[i]:]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict): return candidate
        except (json.JSONDecodeError, ValueError): continue
    tried = 0
    for i in range(len(closes) - 1, -1, -1):
        if tried >= max_candidates: break
        tried += 1
        candidate = text[: closes[i] + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict): return candidate
        except (json.JSONDecodeError, ValueError): continue
    return None


def _parse_tool_input(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return {}
    trimmed = raw.strip()
    try:
        parsed = json.loads(trimmed)
        if isinstance(parsed, dict): return parsed
    except (json.JSONDecodeError, ValueError): pass
    repaired = _last_valid_json_object(trimmed)
    if repaired is not None:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict): return parsed
        except (json.JSONDecodeError, ValueError): pass
    return None


# ---------------------------------------------------------------------------
# Stream State Machine
# ---------------------------------------------------------------------------

class StreamRepairState:
    def __init__(self) -> None:
        self.pending_message_start: dict[str, Any] | None = None
        self.committed = False
        self.open_msg_id: str | None = None
        self.finished = False
        self.failed = False
        self.saw_stop_reason = False

        self.pending_tool_events: list[dict[str, Any]] | None = None
        self.pending_tool_index: int | None = None
        self.seen_block_indices: set[int] = set()
        self.active_block_indices: set[int] = set()
        self._tool_flushed_ok = False

    def process(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or self.failed:
            return []
        etype = payload.get("type")

        # --- Tool block buffering ---
        if self.pending_tool_events is not None:
            if etype == "content_block_delta":
                if payload.get("index") != self.pending_tool_index:
                    return self._fail("tool block index changed")
                self.pending_tool_events.append(payload)
                return []
            if etype == "content_block_stop":
                if payload.get("index") != self.pending_tool_index:
                    return self._fail("tool stop index mismatch")
                self.pending_tool_events.append(payload)
                return self._flush_tool_block()
            return self._fail(f"unexpected {etype!r} before tool block stopped")

        if self.finished and etype != "message_start":
            return []
        if self.finished and etype == "message_start":
            self.finished = False

        # --- Message Lifecycle ---
        if etype == "message_start":
            _repair_message_start(payload)
            new_id = payload["message"]["id"]
            
            out: list[dict[str, Any]] = []
            if self.open_msg_id is not None:
                if new_id == self.open_msg_id:
                    return []  # dup
                # Upstream opened a new message without closing the old one.
                # Force close the old one downstream.
                out.extend(self._close_open_message_forcibly())
            
            self.open_msg_id = new_id
            self.saw_stop_reason = False
            self.seen_block_indices.clear()
            self.active_block_indices.clear()
            self._tool_flushed_ok = False

            if not self.committed:
                # Buffer the start until we see an actual content block
                self.pending_message_start = payload
                return []
            
            out.append(payload)
            return out

        if etype == "error":
            self.failed = True
            return [payload]

        if self.open_msg_id is None:
            return []

        # --- Content Blocks ---
        if etype == "content_block_start":
            idx = payload.get("index")
            block = payload.get("content_block")
            if not isinstance(idx, int) or idx < 0 or not isinstance(block, dict):
                return self._fail("content_block_start invalid")
            if idx in self.seen_block_indices:
                return self._fail(f"duplicate content block index {idx}")
            
            self.seen_block_indices.add(idx)
            self.active_block_indices.add(idx)

            if block.get("type") == "thinking" and not isinstance(block.get("thinking"), str):
                block["thinking"] = ""

            out: list[dict[str, Any]] = []
            if not self.committed:
                if self.pending_message_start is not None:
                    out.append(self.pending_message_start)
                    self.pending_message_start = None
                self.committed = True

            if block.get("type") == "tool_use":
                if not isinstance(block.get("id"), str) or not block["id"]:
                    block["id"] = f"toolu_{uuid.uuid4().hex}"
                if not isinstance(block.get("name"), str) or not block["name"]:
                    return self._fail(f"tool_use at {idx} missing name")
                self.pending_tool_events = [payload]
                self.pending_tool_index = idx
                # Note: we do NOT return [payload] here because the tool block 
                # is buffered. But we MUST return the message_start if we just committed.
                return out

            out.append(payload)
            return out

        if etype == "content_block_delta":
            idx = payload.get("index")
            if idx not in self.active_block_indices:
                return []
            return [payload]

        if etype == "content_block_stop":
            idx = payload.get("index")
            if idx in self.active_block_indices:
                self.active_block_indices.remove(idx)
                return [payload]
            return []

        if etype == "message_delta":
            delta = payload.get("delta")
            if not isinstance(delta, dict):
                return []
            if delta.get("stop_reason") is not None:
                self.saw_stop_reason = True
            return [payload]

        if etype == "message_stop":
            self.open_msg_id = None
            self.finished = True
            return [payload]

        return [payload]

    def finalize(self) -> list[dict[str, Any]]:
        """Flush any dangling state and guarantee a closed stream."""
        out: list[dict[str, Any]] = []

        if self.pending_tool_events is not None:
            out.extend(self._flush_tool_block(allow_missing_stop=True))

        # If we have a buffered start but never committed it, emit an empty message.
        if not self.committed and self.pending_message_start is not None:
            out.append(self.pending_message_start)
            out.append({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
            out.append({"type": "content_block_stop", "index": 0})
            self.pending_message_start = None
            self.committed = True

        # If a message is still open, close it now.
        if self.open_msg_id is not None:
            if not self.saw_stop_reason:
                out.append({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                })
            out.append({"type": "message_stop"})
            self.open_msg_id = None
            self.finished = True

        return out

    def _close_open_message_forcibly(self) -> list[dict[str, Any]]:
        if self.open_msg_id is None: return []
        out: list[dict[str, Any]] = []
        if self.pending_tool_events is not None:
            out.extend(self._flush_tool_block(allow_missing_stop=True))
        
        if not self.saw_stop_reason:
            out.append({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 10, "output_tokens": 10},
            })
        out.append({"type": "message_stop"})
        
        self.open_msg_id = None
        self.active_block_indices.clear()
        self.seen_block_indices.clear()
        return out

    def _fail(self, msg: str) -> list[dict[str, Any]]:
        self.failed = True
        logger.error("Stream protocol error: %s", msg)
        return [{"type": "error", "error": {"type": "api_error", "message": msg}}]

    def _flush_tool_block(self, *, allow_missing_stop: bool = False) -> list[dict[str, Any]]:
        events = self.pending_tool_events or []
        self.pending_tool_events = None
        self.pending_tool_index = None

        start_event = next((e for e in events if e.get("type") == "content_block_start"), None)
        stop_event = next((e for e in events if e.get("type") == "content_block_stop"), None)
        
        if start_event is None:
            return self._fail("tool block missing start")

        idx = start_event["index"]
        block = start_event["content_block"]
        partials: list[str] = []
        for ev in events:
            if ev.get("type") == "content_block_delta":
                d = ev.get("delta")
                if isinstance(d, dict) and d.get("type") == "input_json_delta":
                    p = d.get("partial_json", "")
                    if isinstance(p, str): partials.append(p)

        raw = "".join(partials)
        parsed = _parse_tool_input(raw)
        if parsed is None and isinstance(block.get("input"), dict):
            parsed = block["input"]
        if parsed is None:
            parsed = {}

        new_block = dict(block)
        new_block["input"] = {}
        self.active_block_indices.discard(idx)
        self._tool_flushed_ok = True
        
        if stop_event is None:
            stop_event = {"type": "content_block_stop", "index": idx}

        return [
            {"type": "content_block_start", "index": idx, "content_block": new_block},
            {"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))}},
            stop_event,
        ]


# ---------------------------------------------------------------------------
# SSE Parser & Worker
# ---------------------------------------------------------------------------

def _parse_sse_events(chunks: Iterator[bytes]) -> Iterator[tuple[str | None, str]]:
    buffer = ""
    current_event: str | None = None
    data_lines: list[str] = []

    def dispatch():
        nonlocal current_event, data_lines
        if not data_lines and current_event is None: return None
        item = (current_event, "\n".join(data_lines))
        current_event = None
        data_lines = []
        return item

    for chunk in chunks:
        if not chunk: continue
        buffer += chunk.decode("utf-8", errors="replace")
        while True:
            line_end = None
            for i, ch in enumerate(buffer):
                if ch in "\r\n":
                    if ch == "\r" and i + 1 == len(buffer):
                        line_end = None
                        break
                    line_end = i
                    break
            if line_end is None: break
            line = buffer[:line_end]
            delim = 2 if buffer[line_end:line_end + 2] == "\r\n" else 1
            buffer = buffer[line_end + delim:]
            if line == "":
                ev = dispatch()
                if ev is not None: yield ev
                continue
            if line.startswith(":"): continue
            field, sep, value = line.partition(":")
            if not sep: value = ""
            elif value.startswith(" "): value = value[1:]
            if field == "event": current_event = value
            elif field == "data": data_lines.append(value)

    if buffer:
        line = buffer.rstrip("\r\n")
        if line.startswith("event:"): current_event = line[6:].lstrip(" ")
        elif line.startswith("data:"): data_lines.append(line[5:].lstrip(" "))
    ev = dispatch()
    if ev is not None: yield ev


def _sse_frame(payload: dict[str, Any]) -> bytes:
    et = payload.get("type")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if isinstance(et, str) and et:
        return f"event: {et}\ndata: {data}\n\n".encode()
    return f"data: {data}\n\n".encode()


def _stream_worker(kw: dict[str, Any], out_q: queue.Queue, request: Request) -> None:
    state = StreamRepairState()
    try:
        client = _get_client(request)
        with client.messages.with_streaming_response.create(**kw) as resp:
            for event_name, raw_data in _parse_sse_events(resp.iter_bytes(chunk_size=4096)):
                if event_name == "billing_summary": continue
                if not raw_data: continue
                try:
                    payload = json.loads(raw_data)
                except (json.JSONDecodeError, ValueError):
                    if raw_data.strip() == "[DONE]": continue
                    raise ValueError("non-JSON payload")
                if not isinstance(payload, dict):
                    raise ValueError("payload not object")
                if not payload.get("type") and event_name:
                    payload["type"] = event_name
                
                for repaired in state.process(payload):
                    out_q.put(_sse_frame(repaired))
                if state.failed:
                    return

        # Stream ended naturally. Finalize to close any dangling frames.
        for tail in state.finalize():
            out_q.put(_sse_frame(tail))
            
    except Exception as exc:
        logger.exception("stream worker failed: %s", exc)
        # Ensure we push an error if we haven't committed, or just try to close gracefully.
        if not state.committed:
            out_q.put(_sse_frame({"type": "error", "error": {"type": "api_error", "message": str(exc)}}))
        else:
            # If we already committed (client is reading), we must send a valid stop to prevent UI hanging.
            out_q.put(_sse_frame({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"input_tokens": 0, "output_tokens": 0}}))
            out_q.put(_sse_frame({"type": "message_stop"}))
    finally:
        out_q.put(None)


async def _stream_gen(kw: dict[str, Any], request: Request):
    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=_stream_worker, args=(kw, q, request), daemon=True)
    t.start()
    loop = asyncio.get_running_loop()
    while True:
        try:
            item = await asyncio.wait_for(loop.run_in_executor(None, q.get), timeout=READ_TIMEOUT + 5)
        except asyncio.TimeoutError:
            yield _sse_frame({"type": "error", "error": {"type": "api_error", "message": "proxy timeout"}})
            return
        if item is None: return
        yield item


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app = FastAPI(title="AgentRouter Anthropic Proxy", version="4.0.0")

@app.get("/health")
async def health():
    return {"status": "ok", "upstream": TARGET}

@app.get("/v1/models")
@app.get("/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

@app.post("/v1/messages")
@app.post("/messages")
async def messages(request: Request):
    try:
        body = await request.json()
        kw = _normalize_request(body)
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(400, {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}})

    if body.get("stream", False):
        kw["stream"] = True

        async def safe():
            async for chunk in _stream_gen(kw, request):
                yield chunk

        return StreamingResponse(
            safe(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no", "connection": "keep-alive"},
        )

    try:
        client = _get_client(request)
        msg = await asyncio.to_thread(lambda: client.messages.create(**kw))
        return Response(content=msg.model_dump_json(), media_type="application/json")
    except anthropic.APIStatusError as exc:
        body_val = getattr(exc, "body", None)
        return JSONResponse(exc.status_code, body_val if isinstance(body_val, dict) else {"type": "error", "error": {"type": "api_error", "message": str(body_val or exc)}})
    except Exception as exc:
        return JSONResponse(502, {"type": "error", "error": {"type": "api_error", "message": str(exc)}})

if __name__ == "__main__":
    logger.info("AgentRouter proxy -> %s", TARGET)
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.environ.get("UVICORN_LOG_LEVEL", "warning").lower())