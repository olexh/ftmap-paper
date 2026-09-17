"""The local model, and the cache that makes a rerun free."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import httpx

from ftmap.config import STAGES
from ftmap.fsutil import json_line, open_private
from ftmap.plan.admissible import Inadmissible, format_hint, prune, strip_fences
from ftmap.plan.prompt import (BINDING_MARKER, EDGE_MARKER, REPAIR_MARKER,
                               SPLIT_MARKER, STRUCTURE_MARKER)


class ModelError(RuntimeError):
    pass


def stage_of(user: str) -> str:
    """Which of the five prompt kinds a request is, read off the prompt."""
    if REPAIR_MARKER in user:
        return "repair"
    if STRUCTURE_MARKER in user:
        return "structure"
    if SPLIT_MARKER in user:
        return "split"
    if EDGE_MARKER in user:
        return "edge"
    if BINDING_MARKER in user:
        return "binding"
    return OTHER


OTHER = "other"


def blank_stages() -> dict[str, dict[str, int]]:
    """One zeroed counter row per stage, in `STAGES` order."""
    return {stage: {"attempts": 0, "hits": 0, "generated": 0, "failures": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "discarded": 0}
            for stage in (*STAGES, OTHER)}


_CACHE_FORMAT = 3


def server_fingerprint(props: dict | None) -> str | None:
    """What `/props` says the server is: build, weights file, context size,
    slots. Or None when the server said nothing usable.
    """
    if not props or not props.get("build_info"):
        return None
    model = props.get("model_alias") or os.path.basename(str(props.get("model_path") or ""))
    slots = props.get("total_slots")
    settings = props.get("default_generation_settings") or {}
    n_ctx = settings.get("n_ctx")
    parts = {"build": props.get("build_info"), "model": model,
             "model_path": os.path.basename(str(props.get("model_path") or "")),
             "n_ctx": n_ctx, "slots": slots}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _key(cfg, system: str, user: str, schema: dict, max_tokens: int,
         server: str | None = None) -> str:
    """Everything that could change the answer, and nothing that could not."""
    blob = json.dumps({
        "v": _CACHE_FORMAT,
        "revision": cfg.model_revision,
        "identity": cfg.model_identity,
        "server": server,
        "temperature": cfg.model_temperature,
        "top_p": 1.0,
        "seed": cfg.model_seed,
        "enable_thinking": cfg.enable_thinking,
        "max_tokens": max_tokens,
        "system": system,
        "user": user,
        "schema": schema,
        **({} if getattr(cfg, "grammar", True) else {"grammar": False}),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LlamaClient:
    def __init__(self, cfg, cache_path: str | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg
        self.cache_path = cache_path
        self.calls = 0
        self.cache_hits = 0
        self.stages: dict[str, dict[str, int]] = blank_stages()
        self.reported_model: str | None = None
        self._cache: dict[str, tuple[Any, str | None]] = {}
        self._cache_servers: dict[str, str | None] = {}
        self._client = httpx.Client(transport=transport, timeout=600.0)
        self._replayed_server: str | None = None
        self.server: str | None = self._fingerprint_server()
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("v") != _CACHE_FORMAT or not rec.get("revision"):
                        continue
                    self._cache[rec["key"]] = (rec["response"], rec.get("model"))
                    self._cache_servers[rec["key"]] = rec.get("server")

    def _fingerprint_server(self) -> str | None:
        try:
            resp = self._client.get(self.cfg.model_url + "/props", timeout=5.0)
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            return server_fingerprint(resp.json())
        except ValueError:
            return None

    def _remember(self, key: str, system: str, user: str, response: Any,
                  model: str | None) -> None:
        self._cache[key] = (response, model)
        self._cache_servers[key] = self.server
        if not self.cache_path:
            return
        with open_private(self.cache_path, append=True) as fh:
            fh.write(json_line(
                {"v": _CACHE_FORMAT, "key": key,
                 "revision": self.cfg.model_revision, "model": model,
                 "server": self.server,
                 "system": system, "user": user, "response": response}))

    def model_provenance(self) -> dict[str, str]:
        """What the run manifest records about where its answers came from."""
        return {"revision": self.cfg.model_revision,
                "producing_model": self.reported_model or "unknown",
                "server": self.server or self._replayed_server or "not contacted"}

    def _require_revision(self) -> None:
        """The one place the declared revision is enforced."""
        if not self.cfg.model_revision:
            raise ModelError(
                "[model] revision is not set. A cache entry is only reusable "
                "if the deployment that produced it would produce it again, "
                "and weights, tokenizer, chat template and serving runtime all "
                "count. Declare the revision in ftmap.toml before a run that "
                "calls the model.")

    def close(self) -> None:
        """Release the HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "LlamaClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def complete(self, system: str, user: str, schema: dict,
                 max_tokens: int | None = None) -> dict:
        stage = stage_of(user)
        row = self.stages[stage]
        row["attempts"] += 1
        try:
            return self._complete(stage, row, system, user, schema, max_tokens)
        except ModelError:
            row["failures"] += 1
            raise

    def _complete(self, stage: str, row: dict, system: str, user: str,
                  schema: dict, max_tokens: int | None) -> dict:
        self._require_revision()
        limit = max_tokens or self.cfg.max_tokens
        key = _key(self.cfg, system, user, schema, limit, self.server)
        if self.server is None and key not in self._cache:
            by_server: dict[str | None, str] = {}
            for candidate, owner in self._cache_servers.items():
                want = by_server.get(owner)
                if want is None:
                    want = by_server[owner] = _key(self.cfg, system, user,
                                                   schema, limit, owner)
                if candidate == want:
                    key = candidate
                    self._replayed_server = self._replayed_server or owner
                    break
        if key in self._cache:
            self.cache_hits += 1
            row["hits"] += 1
            response, model = self._cache[key]
            if model and not self.reported_model:
                self.reported_model = model
            return response

        permitted = (set(self.cfg.live_stages) == set(STAGES) if stage == OTHER
                     else stage in self.cfg.live_stages)
        if not permitted:
            raise ModelError(
                f"cache miss on the {stage} stage, and [model] live_stages "
                f"{list(self.cfg.live_stages)} does not permit generating it. "
                "A replay that reaches the server is not a replay.")
        grammar = getattr(self.cfg, "grammar", True)
        body = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user if grammar
                          else user + format_hint(schema)}],
            "temperature": self.cfg.model_temperature,
            "top_p": 1.0,
            "seed": self.cfg.model_seed,
            "max_tokens": limit,
            "chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking},
        }
        if grammar:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "plan", "strict": True, "schema": schema},
            }
        resp = self._client.post(self.cfg.model_url + "/v1/chat/completions", json=body)
        if resp.status_code != 200:
            raise ModelError(f"llama-server returned {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        self.calls += 1
        row["generated"] += 1
        usage = data.get("usage") or {}
        row["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        row["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        reported = data.get("model")
        if self.cfg.model_identity and reported != self.cfg.model_identity:
            raise ModelError(
                f"the server reports model {reported!r}, but [model] identity "
                f"declares {self.cfg.model_identity!r}. Answers from an "
                "undeclared deployment would be cached and reported under a "
                "revision that did not produce them.")
        if reported:
            self.reported_model = reported
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()
        if not content:
            if msg.get("reasoning_content"):
                raise ModelError(
                    "the model answered into its reasoning channel and left "
                    "content empty. Set enable_thinking = false, or start the "
                    "server with --reasoning-budget 0."
                )
            raise ModelError(f"empty response, finish_reason={choice.get('finish_reason')}")
        if not grammar:
            content = strip_fences(content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelError(f"the model did not return JSON: {exc}; got {content[:200]!r}")
        if not grammar:
            try:
                parsed, discarded = prune(parsed, schema)
            except Inadmissible as exc:
                raise ModelError(f"inadmissible free-form answer: {exc}; got "
                                 f"{content[:300]!r}") from exc
            row["discarded"] += discarded
        self._remember(key, system, user, parsed, reported)
        return parsed
