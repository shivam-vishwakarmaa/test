"""A small multi-backend LLM client with an on-disk, content-addressed cache.

The cache is the reason `make eval` reproduces the headline numbers in minutes
without an API key: every model response is keyed by a hash of the exact request
(model, system, user, schema, temperature) and committed to the repo under
`artifacts/llm_cache/`. Running with `LLM_BACKEND=cached` replays those bytes
and *fails loudly* on a cache miss rather than silently calling out or, worse,
silently returning something different from what produced the reported numbers.

Backends
--------
gemini    : real API calls via the current `google-genai` SDK (needs
            GEMINI_API_KEY). Default backend for this submission -- see
            Decision #16 in docs/DECISIONS.md. Writes to the cache.
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

# USD per 1M tokens (input, output). Sources noted per row; entries left at
# (0.0, 0.0) mean "not hard-coded" rather than "free" -- see the comment below.
PRICING = {
    # Anthropic pricing table, cached 2026-06-24.
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
    # Gemini: deliberately left at (0.0, 0.0) rather than a guessed number.
    # This project's GEMINI_API_KEY is a free-tier AI Studio key (confirmed by
    # a 429 RESOURCE_EXHAUSTED on every pro-tier model it was tried against --
    # see Decision #18), so the real run's cost was $0 in practice. Hardcoding
    # a list price for a paid tier this key never touched would misrepresent
    # what this report's numbers actually cost to produce. Token counts are
    # still tracked in full (`Usage.per_model`) so a reader on a paid key can
    # apply the current published rate themselves.
    "gemini-2.5-flash": (0.0, 0.0),
    "gemini-3.5-flash": (0.0, 0.0),
}


class CacheMiss(RuntimeError):
    """Raised when backend='cached' is asked for a request it has never seen."""


class QuotaExhausted(RuntimeError):
    """Raised when a backend reports a DAILY (not per-minute) quota is used up.

    Distinguished from a plain per-item RuntimeError because the right
    response is different: a transient/per-item failure should be logged and
    skipped so the batch continues; a daily quota hit means every subsequent
    call to this model will fail the same way for the rest of the day, so a
    caller iterating a batch should stop immediately rather than burning
    `max_retries` x backoff on every remaining item for no benefit (this bit
    a live 220-example run during development -- see Decision #18)."""


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
                "LLM_BACKEND=gemini or LLM_BACKEND=anthropic (needs the matching API key), or "
                "check that config/config.yaml matches the one used to build artifacts/llm_cache "
                "-- a changed prompt changes the cache key."
            )

        if self.backend == "anthropic":
            parsed, in_tok, out_tok = self._call_anthropic(
                system, user, schema, model, temperature, max_tokens
            )
        elif self.backend == "gemini":
            parsed, in_tok, out_tok = self._call_gemini(
                system, user, schema, model, temperature, max_tokens
            )
        elif self.backend == "ollama":
            parsed, in_tok, out_tok = self._call_ollama(
                system, user, schema, model, temperature, max_tokens
            )
            model = "ollama:" + model
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

    def _call_gemini(
        self, system: str, user: str, schema: dict, model: str,
        temperature: float, max_tokens: int,
    ) -> tuple[dict, int, int]:
        """Uses `google-genai` (the current, maintained SDK) with the API's
        native `response_schema` enforcement -- not a "please return JSON"
        text instruction. Two things this method exists to get right, both
        found by testing against the live API before this was trusted with a
        220-example run (see Decision #17):

        1. `response_schema` must have `additionalProperties` stripped
           (recursively): Gemini's schema dialect is an OpenAPI subset that
           rejects that JSON-Schema keyword outright with a 400, even though
           every schema in this repo carries it for the Anthropic backend.
        2. `thinking_budget=0` is mandatory, not an optimization. Gemini
           2.5/3.5 "thinking" models spend part of `max_output_tokens` on an
           invisible reasoning trace by default; at this codebase's token
           budgets that reasoning silently eats the JSON output and produces
           an unparseable truncated response. Classification/drafting/judging
           here don't need visible chain-of-thought (the schema already has
           an explicit `rationale`/`reasoning` field for that), so thinking
           is switched off entirely rather than budgeted around.
        """
        from google.genai import errors as genai_errors
        from google.genai import types as genai_types

        if "GEMINI_API_KEY" not in os.environ:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required for backend='gemini' "
                "(set it in .env -- see .env.example)"
            )
        client = self._gemini_client()
        clean_schema = _strip_unsupported_schema_keys(schema)

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=user,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        response_schema=clean_schema,
                        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                        # No tools are ever passed, so automatic function calling has
                        # nothing to do here -- disabling it just silences the SDK's
                        # unconditional "consider using Chat instead" advisory log line.
                        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                if not resp.candidates:
                    raise RuntimeError(f"gemini returned no candidates: {resp.prompt_feedback}")
                parsed = _fill_schema_defaults(json.loads(resp.text), schema)
                usage = resp.usage_metadata
                return parsed, usage.prompt_token_count or 0, usage.candidates_token_count or 0
            except genai_errors.ServerError as exc:
                last = exc  # 5xx: transient, worth retrying
                time.sleep(min(2 ** attempt, 30))
            except genai_errors.ClientError as exc:
                last = exc
                if exc.code == 429:
                    if "PerDay" in str(exc):  # daily quota, not a per-minute rate limit -- retrying is futile
                        raise QuotaExhausted(
                            f"gemini backend hit a DAILY quota limit on model={model!r}: {exc}\n"
                            "Retrying will not help until the quota resets (or a paid tier is enabled). "
                            "See docs/DECISIONS.md #18."
                        ) from exc
                    time.sleep(min(2 ** attempt, 30))  # per-minute rate limit: worth backing off and retrying
                else:  # 400/403/404 etc: a real bug (bad schema, bad model name) -- retrying won't help
                    raise RuntimeError(f"gemini rejected the request (non-retryable): {exc}") from exc
            except json.JSONDecodeError as exc:
                last = exc
                time.sleep(1)
        raise RuntimeError(
            "gemini call failed after " + str(self.max_retries) + " attempts: " + str(last)
        )

    def _gemini_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    def _call_ollama(
        self, system: str, user: str, schema: dict, model: str,
        temperature: float, max_tokens: int,
    ) -> tuple[dict, int, int]:
        import urllib.error
        import urllib.request

        model_name = model[7:] if model.startswith("ollama:") else model
        payload = json.dumps({
            "model": model_name,
            "prompt": user,
            "system": system,
            "stream": False,
            "format": schema,  # Ollama supports JSON-schema-constrained output directly
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            self.ollama_url + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read())
                parsed = _fill_schema_defaults(_parse_possibly_fenced_json(data["response"]), schema)
                return parsed, data.get("prompt_eval_count", 0), data.get("eval_count", 0)
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
                time.sleep(min(2 ** attempt, 30))
            except json.JSONDecodeError as exc:
                last = exc
                time.sleep(1)
        raise RuntimeError(
            "ollama call failed after " + str(self.max_retries) + " attempts "
            "(is `ollama serve` running at " + self.ollama_url + "?): " + str(last)
        )


def _strip_unsupported_schema_keys(schema: Any) -> Any:
    """Recursively drop JSON-Schema keywords Gemini's `response_schema`
    rejects (`additionalProperties`, `$schema`, ...). Every schema in this
    codebase is authored once, for the Anthropic `json_schema` output format;
    this keeps it that way instead of maintaining a parallel Gemini copy."""
    _DROP = {"additionalProperties", "$schema"}
    if isinstance(schema, dict):
        return {
            k: _strip_unsupported_schema_keys(v)
            for k, v in schema.items()
            if k not in _DROP
        }
    if isinstance(schema, list):
        return [_strip_unsupported_schema_keys(v) for v in schema]
    return schema


def _parse_possibly_fenced_json(text: str) -> dict:
    """Local/open-weight models (via Ollama) sometimes wrap JSON in a
    markdown code fence despite being asked not to; the hosted backends
    (Anthropic's json_schema mode, Gemini's response_schema) never do."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    return json.loads(text)


def _fill_schema_defaults(parsed: dict, schema: dict) -> dict:
    """Defensive backstop for backends without a hard schema-conformance
    guarantee (Ollama's `format` param is best-effort; a truncated Gemini
    response could in principle still omit a field): fill any missing
    `required` key with a type-appropriate zero value rather than letting a
    downstream `KeyError` crash a 220-example batch run over one bad item."""
    _DEFAULTS = {"array": [], "string": "", "boolean": False, "number": 0.0, "integer": 0}
    for req in schema.get("required", []):
        if req not in parsed:
            prop_type = schema.get("properties", {}).get(req, {}).get("type", "string")
            parsed[req] = _DEFAULTS.get(prop_type, "")
    return parsed


def client_from_config(cfg) -> LLMClient:
    return LLMClient(
        backend=cfg["llm"]["backend"],
        cache_dir=cfg["llm"]["cache_dir"],
        max_retries=int(cfg["llm"].get("max_retries", 4)),
        timeout_s=float(cfg["llm"].get("timeout_s", 120)),
        ollama_url=cfg["llm"].get("ollama_url", "http://localhost:11434"),
        ollama_model=cfg["llm"].get("ollama_model", "qwen2.5:3b"),
    )

