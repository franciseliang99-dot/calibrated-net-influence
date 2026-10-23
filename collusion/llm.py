"""Minimal chat-completions client for the Kimi (Moonshot) OpenAI-compatible API.

The API key is read from the macOS login Keychain at call time and kept in memory
only. It is never printed, logged, or written to disk.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
KEYCHAIN_SERVICE = "KIMI_API_KEY"

_key_cache: str | None = None

# The account is limited to 100 requests per minute across the whole organisation, so every request
# start is spaced at least 60 / KIMI_RPM seconds apart within this process. Run one process at a time.
_RPM = float(os.environ.get("KIMI_RPM", "90"))
_rate_lock = threading.Lock()
_next_slot = 0.0


def _wait_for_slot() -> None:
    global _next_slot
    with _rate_lock:
        now = time.monotonic()
        start = max(now, _next_slot)
        _next_slot = start + 60.0 / _RPM
    time.sleep(max(0.0, start - now))


def _api_key() -> str:
    global _key_cache
    if _key_cache is None:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE,
             "-a", os.environ.get("USER", ""), "-w"],
            capture_output=True, text=True, check=True,
        )
        _key_cache = out.stdout.strip()
        if not _key_cache:
            raise RuntimeError("empty API key from Keychain")
    return _key_cache


def chat(messages: list[dict], model: str = "kimi-k2.6", temperature: float = 1.0,
         max_tokens: int = 512, retries: int = 6, timeout: float = 120.0,
         extra: dict | None = None) -> dict:
    """Return {"text", "usage", "model", "latency_s", "raw_finish"}."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **(extra or {}),
    }).encode()
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {_api_key()}"},
        )
        _wait_for_slot()
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
            if e.code in (429, 500, 502, 503, 504):
                print(f"[llm] HTTP {e.code} on attempt {attempt + 1}, backing off", file=sys.stderr, flush=True)
                time.sleep(2 ** attempt * 2)
                continue
            raise last_err from None
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            print(f"[llm] {type(e).__name__} on attempt {attempt + 1}, backing off", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt * 2)
            continue
        choice = data["choices"][0]
        msg = choice["message"]
        return {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content"),
            "usage": data.get("usage", {}),
            "model": data.get("model", model),
            "latency_s": round(time.monotonic() - t0, 2),
            "raw_finish": choice.get("finish_reason"),
        }
    raise RuntimeError(f"chat failed after {retries} attempts: {last_err}")
