"""What a grammar would have refused, refused after the fact."""
from __future__ import annotations

import json
from typing import Any

import jsonschema


class Inadmissible(ValueError):
    """The answer cannot be made admissible by discarding entries."""


_INVALID = object()


def _prune(value: Any, schema: dict, dropped: list[int]) -> Any:
    """`value` with every inadmissible entry removed, or `_INVALID`."""
    if "const" in schema:
        return value if value == schema["const"] else _INVALID
    if "enum" in schema:
        return value if value in schema["enum"] else _INVALID
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return _INVALID
        props = schema.get("properties") or {}
        out: dict = {}
        for name, sub in props.items():
            if name not in value:
                continue
            got = _prune(value[name], sub, dropped)
            if got is _INVALID:
                if name in (schema.get("required") or ()):
                    return _INVALID
                dropped[0] += 1
                continue
            out[name] = got
        for name in schema.get("required") or ():
            if name not in out:
                return _INVALID
        if schema.get("additionalProperties", True) is not False:
            for name, v in value.items():
                if name not in props:
                    out[name] = v
        else:
            dropped[0] += sum(1 for name in value if name not in props)
        return out
    if kind == "array":
        if not isinstance(value, list):
            return _INVALID
        branches = schema.get("prefixItems")
        item_schema = schema.get("items")
        out_list: list = []
        for item in value:
            kept = _INVALID
            if branches:
                for branch in branches:
                    kept = _prune(item, branch, dropped)
                    if kept is not _INVALID:
                        break
            elif item_schema:
                kept = _prune(item, item_schema, dropped)
            else:
                kept = item
            if kept is _INVALID:
                dropped[0] += 1
                continue
            out_list.append(kept)
        return out_list
    if kind == "string":
        if not isinstance(value, str):
            return _INVALID
        limit = schema.get("maxLength")
        if limit is not None and len(value) > limit:
            value = value[:limit]
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError:
        return _INVALID
    return value


def _relaxed(schema: dict) -> dict:
    """The schema with the positional and length constraints the pruning
    does not enforce removed, for the final check."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in ("minItems", "maxItems"):
                continue
            if k == "prefixItems":
                out["items"] = {"anyOf": [_relaxed(b) for b in v]}
                continue
            out[k] = _relaxed(v)
        return out
    if isinstance(schema, list):
        return [_relaxed(x) for x in schema]
    return schema


def prune(value: Any, schema: dict) -> tuple[Any, int]:
    """`(admissible answer, entries discarded)`. Raises `Inadmissible` when
    no pruning makes the answer admissible."""
    dropped = [0]
    got = _prune(value, schema, dropped)
    if got is _INVALID:
        raise Inadmissible("the answer is not admissible under the response "
                           "schema even after discarding entries")
    try:
        jsonschema.validate(got, _relaxed(schema))
    except jsonschema.ValidationError as exc:
        raise Inadmissible(f"pruned answer still fails the schema: "
                           f"{exc.message}") from exc
    return got, dropped[0]


def strip_fences(content: str) -> str:
    """A free answer may wrap its JSON in a Markdown code fence."""
    text = content.strip()
    if text.startswith("```"):
        first = text.find("\n")
        text = text[first + 1:] if first >= 0 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def format_hint(schema: dict) -> str:
    """What the prompt says instead of the grammar."""
    return ("\n\nAnswer with one JSON object and nothing else — no prose, no "
            "code fence. It must satisfy this JSON Schema exactly; every "
            "string value must be one of the choices the schema lists:\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
