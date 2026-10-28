#!/usr/bin/env python3
"""
AgentRouter Anthropic-compatible proxy for OpenCode.

- Tolerates reused content_block indices (no hard fail on duplicates)
- Emits tool_use start immediately; repairs tool JSON at stop
- Forces stop_reason=tool_use when a tool was emitted
- Maps upstream 502/503/504 HTML errors to short client messages
- Clear stdout banner on startup; httpx2 with httpx fallback
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator

import anthropic
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

try:
    import httpx2 as _httpx
except ImportError:
    import httpx as _httpx

TARGET = os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org").rstrip("/")
HOST = os.environ.get("AGENTROUTER_HOST", "127.0.0.1")
PORT = int(os.environ.get("AGENTROUTER_PORT", "7187"))
KEY_FILE = Path.home() / ".config/opencode/api_keys/AGENT_ROUTER_API_KEY"
CONNECT_TIMEOUT = float(os.environ.get("AGENTROUTER_CONNECT_TIMEOUT", "15"))
READ_TIMEOUT = float(os.environ.get("AGENTROUTER_READ_TIMEOUT", "900"))
WRITE_TIMEOUT = float(os.environ.get("AGENTROUTER_WRITE_TIMEOUT", "30"))
POOL_TIMEOUT = float(os.environ.get("AGENTROUTER_POOL_TIMEOUT", "10"))
MODEL_ID = os.environ.get("AGENTROUTER_MODEL_ID", "gpt-5.6-sol")
KEEP_CACHE_CONTROL = os.environ.get("AGENTROUTER_KEEP_CACHE_CONTROL", "0") == "1"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)
logger = logging.getLogger("agentrouter-proxy")

_HTTP_CLIENT = _httpx.Client(
    limits=_httpx.Limits(
        max_connections=int(os.environ.get("AGENTROUTER_MAX_CONNECTIONS", "50")),
        max_keepalive_connections=int(os.environ.get("AGENTROUTER_MAX_KEEPALIVE", "10")),
        keepalive_expiry=20.0,
    ),
    timeout=_httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT,
    ),
)

_app_client_lock = threading.Lock()
_anthropic_client: anthropic.Anthropic | None = None

_SDK_FIELDS = {
    "model", "messages", "system", "max_tokens", "metadata", "stop_sequences",
    "temperature", "top_k", "top_p", "tools", "tool_choice", "service_tier",
}
_HTML_RE = re.compile(r"<[^>]+>")


class ProxyConfigurationError(RuntimeError):
    pass


def _read_server_key() -> str | None:
    key = os.environ.get("AGENTROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        if KEY_FILE.exists():
            text = KEY_FILE.read_text(encoding="utf-8").strip()
            if text:
                return text
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


def _friendly_upstream_error(exc: BaseException) -> str:
    """Turn ALB HTML / SDK noise into a short message for OpenCode."""
    status = getattr(exc, "status_code", None)
    text = str(exc) or type(exc).__name__
    low = text.lower()
    body = getattr(exc, "body", None)
    if isinstance(body, (bytes, bytearray)):
        try:
            text = body.decode("utf-8", errors="replace")
            low = text.lower()
        except Exception:
            pass
    elif isinstance(body, str) and body:
        text = body
        low = text.lower()

    if status == 504 or "504" in text or "gateway time-out" in low or "gateway timeout" in low:
        return (
            "Upstream AgentRouter timed out (504 Gateway Time-out). "
            "Retry later, shorten the conversation context, or try another model."
        )
    if status == 502 or "502" in text or "bad gateway" in low:
        return "Upstream AgentRouter bad gateway (502). Retry later."
    if status == 503 or "503" in text or "service unavailable" in low:
        return "Upstream AgentRouter unavailable (503). Retry later."
    if status == 529 or "overloaded" in low:
        return "Upstream model is overloaded. Retry later."
    if "<html" in low or "<center>" in low:
        plain = _HTML_RE.sub(" ", text)
        plain = re.sub(r"\s+", " ", plain).strip()
        return (plain[:240] if plain else "Upstream returned an HTML error page.")
    if isinstance(status, int) and status >= 400:
        return f"Upstream HTTP {status}: {text[:300]}"
    return text[:500]


def _strip_block_cache(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    out: dict[str, Any] = {}
    for key, value in block.items():
        if key == "cache_control":
            continue
        if key == "content" and isinstance(value, list):
            out[key] = [
                _strip_block_cache(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            out[key] = value
    return out


def _strip_cache_control(body: dict[str, Any]) -> None:
    system = body.get("system")
    if isinstance(system, list):
        body["system"] = [
            _strip_block_cache(item) if isinstance(item, dict) else item for item in system
        ]
    tools = body.get("tools")
    if isinstance(tools, list):
        body["tools"] = [
            _strip_block_cache(item) if isinstance(item, dict) else item for item in tools
        ]
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                _strip_block_cache(item) if isinstance(item, dict) else item
                for item in content
            ]
        cleaned.append(msg)
    body["messages"] = cleaned


def _normalize_thinking(thinking: Any) -> dict[str, Any] | None:
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return None
    budget = thinking.get("budget_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _normalize_request(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("Request JSON body must be an object")
    if not isinstance(body.get("model"), str) or not body["model"].strip():
        raise ValueError("Request field 'model' must be a non-empty string")
    if not isinstance(body.get("messages"), list):
        raise ValueError("Request field 'messages' must be an array")
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("Request field 'max_tokens' must be a positive integer")
    allow_unknown = os.environ.get("AGENTROUTER_ALLOW_UNKNOWN_FIELDS", "0") == "1"
    if allow_unknown:
        result = {
            k: v for k, v in body.items() if k not in {"stream", "betas", "output_config"}
        }
    else:
        result = {k: v for k, v in body.items() if k in _SDK_FIELDS and v is not None}
    thinking = _normalize_thinking(body.get("thinking"))
    if thinking is not None:
        result["thinking"] = thinking
    if not KEEP_CACHE_CONTROL:
        _strip_cache_control(result)
    return result


def _extra_headers(request: Request, body: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    version = request.headers.get("anthropic-version")
    if version:
        headers["anthropic-version"] = version
    betas: list[str] = []
    header_beta = request.headers.get("anthropic-beta")
    if header_beta:
        betas.extend(part.strip() for part in header_beta.split(",") if part.strip())
    raw = body.get("betas")
    if isinstance(raw, str):
        betas.extend(part.strip() for part in raw.split(",") if part.strip())
    elif isinstance(raw, list):
        betas.extend(item.strip() for item in raw if isinstance(item, str) and item.strip())
    ordered: list[str] = []
    seen: set[str] = set()
    for beta in betas:
        if beta not in seen:
            seen.add(beta)
            ordered.append(beta)
    if ordered:
        headers["anthropic-beta"] = ",".join(ordered)
    return headers


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _coerce_token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _repair_message_start(payload: dict[str, Any], model: str) -> None:
    msg = payload.get("message")
    if not isinstance(msg, dict):
        msg = {}
        payload["message"] = msg
    if not isinstance(msg.get("id"), str) or not msg["id"]:
        msg["id"] = _new_message_id()
    msg["type"] = "message"
    msg["role"] = "assistant"
    msg["content"] = []
    if not isinstance(msg.get("model"), str) or not msg["model"]:
        msg["model"] = model
    usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
    msg["usage"] = {
        "input_tokens": _coerce_token_count(usage.get("input_tokens")),
        "output_tokens": _coerce_token_count(usage.get("output_tokens")),
    }
    msg["stop_reason"] = None
    msg["stop_sequence"] = None


def _json_points(text: str) -> list[tuple[int, str]]:
    stack: list[str] = []
    points: list[tuple[int, str]] = []
    in_str = False
    escape = False
    index = 0
    limit = len(text)

    def closers() -> str:
        return "".join(reversed(stack))

    while index < limit:
        ch = text[index]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
                look = index + 1
                while look < limit and text[look] in " \t\r\n":
                    look += 1
                if look >= limit or text[look] != ":":
                    points.append((index + 1, closers()))
            index += 1
            continue
        if ch == '"':
            in_str = True
            index += 1
            continue
        if ch == "{":
            stack.append("}")
            index += 1
            continue
        if ch == "[":
            stack.append("]")
            index += 1
            continue
        if ch in "}]":
            if not stack or stack[-1] != ch:
                break
            stack.pop()
            points.append((index + 1, closers()))
            index += 1
            continue
        literal_end = None
        if text.startswith("true", index):
            literal_end = index + 4
        elif text.startswith("false", index):
            literal_end = index + 5
        elif text.startswith("null", index):
            literal_end = index + 4
        elif ch == "-" or ch.isdigit():
            literal_end = index + 1
            while literal_end < limit and text[literal_end] in "0123456789+-.eE":
                literal_end += 1
        if literal_end is not None:
            points.append((literal_end, closers()))
            index = literal_end
            continue
        index += 1
    return points


def _best_json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if not raw:
        return None
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    points = _json_points(raw)
    if not points:
        return None
    end, closers = points[-1]
    if raw[end:].strip("` \t\r\n,}]"):
        return None
    candidate = raw[:end].rstrip().rstrip(",") + closers
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) and parsed else None


def _json_object(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class StreamRepairState:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or MODEL_ID
        self.started = False
        self.finished = False
        self.failed = False
        self.any_block_emitted = False
        self.emitted_tool = False
        self.stop_reason: str | None = None
        self.stop_sequence: str | None = None
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.seen_block_indices: set[int] = set()
        self.active_blocks: dict[int, str] = {}
        self.index_remap: dict[int, int] = {}
        self.tool_buf: dict[str, Any] | None = None

    def process(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or self.failed:
            return []
        out: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = payload
        for _ in range(8):
            if pending is None:
                break
            events, pending = self._step(pending)
            out.extend(events)
        return out

    def finalize(self) -> list[dict[str, Any]]:
        if self.failed or self.finished:
            return []
        return self._close_message()

    def _step(
        self, payload: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if self.finished or self.failed:
            return [], None
        etype = payload.get("type")

        if self.tool_buf is not None:
            if etype == "ping":
                return [{"type": "ping"}], None
            if not self._is_tool_continuation(payload):
                return self._flush_tool(), payload
            return self._buffer_tool_event(payload), None

        if etype == "ping":
            return ([{"type": "ping"}], None) if self.started else ([], None)
        if etype == "error":
            return self._on_error(payload), None
        if etype == "message_start":
            if self.started:
                logger.warning("dropping extra message_start")
                return self._close_message(), None
            _repair_message_start(payload, self.model)
            self.started = True
            self._merge_usage(payload["message"].get("usage"))
            return [payload], None
        if not self.started:
            if etype in {
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            }:
                return self._ensure_started(), payload
            return [], None
        if etype == "message_delta":
            return self._on_message_delta(payload), None
        if etype == "message_stop":
            return self._close_message(), None
        if etype == "content_block_start":
            return self._on_block_start(payload), None
        if etype == "content_block_delta":
            return self._on_block_delta(payload), None
        if etype == "content_block_stop":
            return self._on_block_stop(payload), None
        return [], None

    def _ensure_started(self) -> list[dict[str, Any]]:
        if self.started:
            return []
        payload: dict[str, Any] = {"type": "message_start", "message": {}}
        _repair_message_start(payload, self.model)
        self.started = True
        return [payload]

    def _merge_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        for key in ("input_tokens", "output_tokens"):
            if key in usage:
                self.usage[key] = _coerce_token_count(usage.get(key))

    def _final_stop_reason(self) -> str:
        if self.emitted_tool and self.stop_reason in (None, "end_turn"):
            return "tool_use"
        if isinstance(self.stop_reason, str) and self.stop_reason:
            return self.stop_reason
        return "end_turn"

    def _close_open_blocks(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for index in sorted(self.active_blocks):
            out.append({"type": "content_block_stop", "index": index})
        self.active_blocks.clear()
        return out

    def _close_message(self) -> list[dict[str, Any]]:
        if self.finished:
            return []
        out = self._ensure_started()
        if self.tool_buf is not None:
            out.extend(self._flush_tool())
        out.extend(self._close_open_blocks())
        if not self.any_block_emitted:
            out.append(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            out.append({"type": "content_block_stop", "index": 0})
            self.any_block_emitted = True
        out.append(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._final_stop_reason(),
                    "stop_sequence": self.stop_sequence,
                },
                "usage": {
                    "input_tokens": self.usage.get("input_tokens", 0),
                    "output_tokens": self.usage.get("output_tokens", 0),
                },
            }
        )
        out.append({"type": "message_stop"})
        self.finished = True
        return out

    def _on_error(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if self.started and self.any_block_emitted:
            logger.error(
                "upstream error after content, closing message: %s", payload.get("error")
            )
            return self._close_message()
        self.failed = True
        err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        return [
            {
                "type": "error",
                "error": {
                    "type": str(err.get("type") or "api_error"),
                    "message": str(err.get("message") or "upstream error"),
                },
            }
        ]

    def _is_tool_continuation(self, payload: dict[str, Any]) -> bool:
        if self.tool_buf is None:
            return False
        etype = payload.get("type")
        if etype not in {"content_block_delta", "content_block_stop"}:
            return False
        upstream = payload.get("index")
        idx = self.index_remap.get(upstream, upstream)
        return idx == self.tool_buf["index"]

    def _on_block_start(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        upstream = payload.get("index")
        block = payload.get("content_block")
        if (
            not isinstance(upstream, int)
            or isinstance(upstream, bool)
            or upstream < 0
            or not isinstance(block, dict)
        ):
            return []

        out: list[dict[str, Any]] = []
        mapped = self.index_remap.get(upstream)
        if mapped is not None and mapped in self.active_blocks:
            logger.warning(
                "force-closing open content block index %s before reuse", mapped
            )
            out.append({"type": "content_block_stop", "index": mapped})
            self.active_blocks.pop(mapped, None)
            idx = mapped
        elif mapped is not None or upstream in self.seen_block_indices:
            idx = (max(self.seen_block_indices) + 1) if self.seen_block_indices else 0
            logger.warning(
                "remapping reused content block index %s -> %s", upstream, idx
            )
        else:
            idx = upstream

        self.index_remap[upstream] = idx
        self.seen_block_indices.add(idx)
        payload = dict(payload)
        payload["index"] = idx

        block_type = block.get("type")
        if block_type not in {"text", "thinking", "redacted_thinking", "tool_use"}:
            return out

        if block_type == "tool_use":
            tool_id = (
                block.get("id")
                if isinstance(block.get("id"), str) and block.get("id")
                else f"toolu_{uuid.uuid4().hex}"
            )
            tool_name = block.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                return out
            fallback = block.get("input")
            if isinstance(fallback, str):
                fallback = _best_json_object(fallback)
            if not isinstance(fallback, dict) or not fallback:
                fallback = None
            self.tool_buf = {"index": idx, "parts": [], "fallback": fallback}
            self.active_blocks[idx] = "tool_use"
            self.any_block_emitted = True
            out.append(
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": {},
                    },
                }
            )
            return out

        if block_type == "text" and not isinstance(block.get("text"), str):
            block["text"] = ""
        if block_type == "thinking" and not isinstance(block.get("thinking"), str):
            block["thinking"] = ""
        if block_type == "redacted_thinking" and not isinstance(block.get("data"), str):
            block["data"] = ""
        self.active_blocks[idx] = str(block_type)
        self.any_block_emitted = True
        out.append(payload)
        return out

    def _buffer_tool_event(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        assert self.tool_buf is not None
        etype = payload.get("type")
        if etype == "content_block_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                part = delta.get("partial_json", "")
                self.tool_buf["parts"].append(
                    part if isinstance(part, str) else str(part)
                )
            return []
        if etype == "content_block_stop":
            return self._flush_tool()
        return []

    def _flush_tool(self) -> list[dict[str, Any]]:
        buf = self.tool_buf
        self.tool_buf = None
        if buf is None:
            return []
        index = buf["index"]
        self.active_blocks.pop(index, None)
        raw = "".join(buf["parts"])
        if raw.strip():
            parsed = _best_json_object(raw)
            partial = _json_object(parsed) if parsed is not None else raw
        elif isinstance(buf.get("fallback"), dict):
            partial = _json_object(buf["fallback"])
        else:
            partial = "{}"
        self.emitted_tool = True
        self.any_block_emitted = True
        return [
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": partial},
            },
            {"type": "content_block_stop", "index": index},
        ]

    def _on_block_delta(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        upstream = payload.get("index")
        idx = self.index_remap.get(upstream, upstream)
        if idx not in self.active_blocks:
            return []
        if idx != upstream:
            payload = dict(payload)
            payload["index"] = idx
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return []
        dtype = delta.get("type")
        if dtype == "text_delta":
            if not isinstance(delta.get("text"), str):
                delta["text"] = (
                    "" if delta.get("text") is None else str(delta.get("text"))
                )
            return [payload]
        if dtype == "thinking_delta":
            if not isinstance(delta.get("thinking"), str):
                delta["thinking"] = (
                    ""
                    if delta.get("thinking") is None
                    else str(delta.get("thinking"))
                )
            return [payload]
        if dtype == "signature_delta" and isinstance(delta.get("signature"), str):
            return [payload]
        if dtype == "input_json_delta":
            if not isinstance(delta.get("partial_json"), str):
                delta["partial_json"] = (
                    ""
                    if delta.get("partial_json") is None
                    else str(delta.get("partial_json"))
                )
            return [payload]
        return []

    def _on_block_stop(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        upstream = payload.get("index")
        idx = self.index_remap.get(upstream, upstream)
        block_type = self.active_blocks.pop(idx, None)
        if block_type is None:
            return []
        if block_type == "tool_use":
            self.emitted_tool = True
        if idx != upstream:
            payload = dict(payload)
            payload["index"] = idx
        return [payload]

    def _on_message_delta(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        delta = payload.get("delta")
        out = self._close_open_blocks()
        if isinstance(delta, dict):
            reason = delta.get("stop_reason")
            if isinstance(reason, str) and reason:
                self.stop_reason = reason
            if "stop_sequence" in delta:
                seq = delta.get("stop_sequence")
                self.stop_sequence = seq if isinstance(seq, str) else None
            if self.emitted_tool and self.stop_reason in (None, "end_turn"):
                self.stop_reason = "tool_use"
        self._merge_usage(payload.get("usage"))
        return out


def _parse_sse_events(chunks: Iterator[bytes]) -> Iterator[tuple[str | None, str]]:
    buffer = ""
    current_event: str | None = None
    data_lines: list[str] = []

    def dispatch():
        nonlocal current_event, data_lines
        if not data_lines and current_event is None:
            return None
        item = (current_event, "\n".join(data_lines))
        current_event = None
        data_lines = []
        return item

    for chunk in chunks:
        if not chunk:
            continue
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
            if line_end is None:
                break
            line = buffer[:line_end]
            delim = 2 if buffer[line_end : line_end + 2] == "\r\n" else 1
            buffer = buffer[line_end + delim :]
            if line == "":
                ev = dispatch()
                if ev is not None:
                    yield ev
                continue
            if line.startswith(":"):
                continue
            field, sep, value = line.partition(":")
            if not sep:
                value = ""
            elif value.startswith(" "):
                value = value[1:]
            if field == "event":
                current_event = value
            elif field == "data":
                data_lines.append(value)
    if buffer:
        line = buffer.rstrip("\r\n")
        if line.startswith("event:"):
            current_event = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    ev = dispatch()
    if ev is not None:
        yield ev


def _sse_frame(payload: dict[str, Any]) -> bytes:
    et = payload.get("type")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if isinstance(et, str) and et:
        return f"event: {et}\ndata: {data}\n\n".encode()
    return f"data: {data}\n\n".encode()


def _handle_upstream_event(
    state: StreamRepairState, event_name: str | None, raw_data: str
) -> list[dict[str, Any]]:
    if event_name == "billing_summary":
        return []
    if not raw_data:
        return state.process({"type": "ping"}) if event_name == "ping" else []
    if raw_data.strip() == "[DONE]":
        return []
    try:
        payload = json.loads(raw_data)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("non-JSON payload")
    if not isinstance(payload, dict):
        raise ValueError("payload not object")
    if not payload.get("type") and event_name:
        payload["type"] = event_name
    if payload.get("type") == "billing_summary":
        return []
    return state.process(payload)


def _stream_worker(
    kw: dict[str, Any], extra_headers: dict[str, str], out_q: queue.Queue
) -> None:
    state = StreamRepairState(
        model=kw.get("model") if isinstance(kw.get("model"), str) else None
    )
    try:
        client = _get_client()
        call_kw = dict(kw)
        if extra_headers:
            call_kw["extra_headers"] = extra_headers
        with client.messages.with_streaming_response.create(**call_kw) as resp:
            for event_name, raw_data in _parse_sse_events(
                resp.iter_bytes(chunk_size=4096)
            ):
                for repaired in _handle_upstream_event(state, event_name, raw_data):
                    out_q.put(_sse_frame(repaired))
                if state.failed:
                    return
        for tail in state.finalize():
            out_q.put(_sse_frame(tail))
    except Exception as exc:
        msg = _friendly_upstream_error(exc)
        logger.error("stream worker failed: %s", msg)
        if not state.failed:
            if not state.started:
                out_q.put(
                    _sse_frame(
                        {
                            "type": "error",
                            "error": {"type": "api_error", "message": msg},
                        }
                    )
                )
            else:
                for tail in state.finalize():
                    out_q.put(_sse_frame(tail))
    finally:
        out_q.put(None)


async def _stream_gen(kw: dict[str, Any], extra_headers: dict[str, str]):
    q: queue.Queue = queue.Queue()
    threading.Thread(
        target=_stream_worker, args=(kw, extra_headers, q), daemon=True
    ).start()
    loop = asyncio.get_running_loop()
    while True:
        try:
            item = await asyncio.wait_for(
                loop.run_in_executor(None, q.get), timeout=READ_TIMEOUT + 5
            )
        except asyncio.TimeoutError:
            yield _sse_frame(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "proxy timeout waiting for upstream stream",
                    },
                }
            )
            return
        if item is None:
            return
        yield item


app = FastAPI(title="AgentRouter Anthropic Proxy", version="5.1.1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "upstream": TARGET,
        "listen": f"{HOST}:{PORT}",
        "model_id": MODEL_ID,
    }


@app.get("/v1/models")
@app.get("/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/messages")
@app.post("/messages")
async def messages(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Request JSON body must be an object")
        extra_headers = _extra_headers(request, body)
        kw = _normalize_request(body)
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            400,
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )

    if body.get("stream", False):
        kw["stream"] = True

        async def safe():
            async for chunk in _stream_gen(kw, extra_headers):
                yield chunk

        return StreamingResponse(
            safe(),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
                "connection": "keep-alive",
            },
        )

    try:
        client = _get_client(request)
        call_kw = dict(kw)
        if extra_headers:
            call_kw["extra_headers"] = extra_headers
        msg = await asyncio.to_thread(lambda: client.messages.create(**call_kw))
        return Response(content=msg.model_dump_json(), media_type="application/json")
    except anthropic.APIStatusError as exc:
        msg = _friendly_upstream_error(exc)
        status = getattr(exc, "status_code", 502) or 502
        return JSONResponse(
            status,
            {"type": "error", "error": {"type": "api_error", "message": msg}},
        )
    except Exception as exc:
        return JSONResponse(
            502,
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": _friendly_upstream_error(exc),
                },
            },
        )


if __name__ == "__main__":
    print(f"AgentRouter proxy starting on http://{HOST}:{PORT}", flush=True)
    print(f"  upstream : {TARGET}", flush=True)
    print(f"  model id : {MODEL_ID}", flush=True)
    print(f"  health   : http://{HOST}:{PORT}/health", flush=True)
    logger.info("AgentRouter proxy -> %s (listen %s:%s)", TARGET, HOST, PORT)
    try:
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level=os.environ.get("UVICORN_LOG_LEVEL", "info").lower(),
        )
    except OSError as exc:
        print(f"Failed to bind {HOST}:{PORT}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
