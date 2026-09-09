"""A small multi-backend LLM client with an on-disk, content-addressed cache.

The cache is the reason `make eval` reproduces the headline numbers in minutes
without an API key: every model response is keyed by a hash of the exact request
(model, system, user, schema, temperature) and committed to the repo under
`artifacts/llm_cache/`. Running with `LLM_BACKEND=cached` replays those bytes
and *fails loudly* on a cache miss rather than silently calling out or, worse,
silently returning something different from what produced the reported numbers.

Backends
--------
anthropic : real API calls (needs ANTHROPIC_API_KEY). Writes to the cache.
cached    : replay only. No network. Raises CacheMiss on an unseen request.
ollama    : local generative model, for people who want zero API spend.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

# USD per 1M tokens, from the Anthropic pricing table (cached 2026-06-24).
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
}


class CacheMiss(RuntimeError):
    """Raised when backend='cached' is asked for a request it has never seen."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, model: str, in_tok: int, out_tok: int) -> None:
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        m = self.per_model.setdefault(model, {"calls": 0, "in": 0, "out": 0})
        m["calls"] += 1
        m["in"] += in_tok
        m["out"] += out_tok

    @property
    def cost_usd(self) -> float:
        total = 0.0
        for model, m in self.per_model.items():
            pin, pout = PRICING.get(model, (0.0, 0.0))
            total += m["in"] / 1e6 * pin + m["out"] / 1e6 * pout
        return total

    def summary(self) -> str:
        return (
            f"{self.calls} calls ({self.cache_hits} cache hits), "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens, "
            f"${self.cost_usd:.4f}"
        )


def _key(model: str, system: str, user: str, schema: Any, temperature: float, tag: str) -> str:
    h = hashlib.sha256()
    payload = json.dumps(
        {
            "model": model, "system": system, "user": user,
            "schema": schema, "temperature": temperature, "tag": tag,
        },
        sort_keys=True, ensure_ascii=False,
    )
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


class LLMClient:
    def __init__(
        self,
        backend: str = "cached",
        cache_dir: str = "artifacts/llm_cache",
        max_retries: int = 4,
        timeout_s: float = 120.0,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:3b",
    ) -> None:
        self.backend = backend
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.usage = Usage()
        os.makedirs(cache_dir, exist_ok=True)
        self._client = None

    # ------------------------------------------------------------------ cache
    def _cache_path(self, key: str) -> str:
        # Shard by first two hex chars so the directory stays browsable.
        sub = os.path.join(self.cache_dir, key[:2])
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, key + ".json")

    def _read_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _write_cache(self, key: str, record: dict) -> None:
        with open(self._cache_path(key), "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1)

    # ----------------------------------------------------------------- public
    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tag: str = "",
    ) -> dict:
        """Return a dict validated against `schema` by the API structured output."""
        key = _key(model, system, user, schema, temperature, tag)
        hit = self._read_cache(key)
        if hit is not None:
            self.usage.cache_hits += 1
            return hit["parsed"]

        if self.backend == "cached":
            raise CacheMiss(
                "No cached response for tag=" + repr(tag) + " model=" + model
                + " key=" + key[:12] + ".\nThis request was never run. Either re-run with "
                "LLM_BACKEND=anthropic (costs money), or check that config/config.yaml "
                "matches the one used to build artifacts/llm_cache -- a changed prompt "
                "changes the cache key."
            )

        if self.backend == "anthropic":
            parsed, in_tok, out_tok = self._call_anthropic(
                system, user, schema, model, temperature, max_tokens
            )
        elif self.backend == "ollama":
            parsed, in_tok, out_tok = self._call_ollama(system, user, schema, max_tokens)
            model = "ollama:" + self.ollama_model
        else:
            raise ValueError("unknown llm backend " + repr(self.backend))

        self.usage.add(model, in_tok, out_tok)
        self._write_cache(key, {
            "tag": tag, "model": model, "system": system, "user": user,
            "schema": schema, "temperature": temperature,
            "parsed": parsed, "input_tokens": in_tok, "output_tokens": out_tok,
            "ts": time.time(),
        })
        return parsed

    # --------------------------------------------------------------- backends
    def _anthropic_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pip install anthropic") from exc
            self._client = anthropic.Anthropic(timeout=self.timeout_s)
        return self._client

    def _call_anthropic(
        self, system: str, user: str, schema: dict, model: str,
        temperature: float, max_tokens: int,
    ) -> tuple[dict, int, int]:
        import anthropic

        client = self._anthropic_client()
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        # Sampling params were removed on the 4.6+ family; only send temperature
        # to models that still accept it.
        if model.startswith("claude-haiku"):
            kwargs["temperature"] = temperature

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(**kwargs)
                if getattr(resp, "stop_reason", None) == "refusal":
                    raise RuntimeError("model refused: " + str(getattr(resp, "stop_details", None)))
                text = next(b.text for b in resp.content if b.type == "text")
                return (
                    json.loads(text),
                    resp.usage.input_tokens,
                    resp.usage.output_tokens,
                )
            except (anthropic.RateLimitError, anthropic.APIConnectionError,
                    anthropic.APITimeoutError, anthropic.InternalServerError) as exc:
                last = exc
                time.sleep(min(2 ** attempt, 30))
            except json.JSONDecodeError as exc:
                last = exc
                time.sleep(1)
        raise RuntimeError(
            "anthropic call failed after " + str(self.max_retries) + " attempts: " + str(last)
        )

    def _call_ollama(
        self, system: str, user: str, schema: dict, max_tokens: int
    ) -> tuple[dict, int, int]:
        import urllib.request

        payload = json.dumps({
            "model": self.ollama_model,
            "prompt": user,
            "system": system,
            "stream": False,
            "format": schema,           # Ollama supports JSON-schema constrained output
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            self.ollama_url + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read())
        return (
            json.loads(data["response"]),
            data.get("prompt_eval_count", 0),
            data.get("eval_count", 0),
        )


def client_from_config(cfg) -> LLMClient:
    return LLMClient(
        backend=cfg.get_path("llm.backend", "cached"),
        cache_dir=cfg.get_path("llm.cache_dir", "artifacts/llm_cache"),
        max_retries=int(cfg.get_path("llm.max_retries", 4)),
        timeout_s=float(cfg.get_path("llm.timeout_s", 120)),
    )
