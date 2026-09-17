"""Run configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any
from ipaddress import ip_address
from urllib.parse import urlparse


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _is_loopback(url: str) -> bool:
    """Whether a model endpoint stays on this machine."""
    host = urlparse(url).hostname or ""
    if not host:
        return False
    if host in {"localhost"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


STAGES = ("structure", "split", "edge", "binding", "repair")

BINDING_MODES = ("model", "heuristic", "heuristic-first", "static", "property")
STRUCTURE_MODES = ("model", "heuristic")
REFINEMENT_MODES = ("rules", "none")


@dataclass(frozen=True)
class Config:
    model_url: str
    model_seed: int
    model_temperature: float
    enable_thinking: bool
    max_tokens: int
    shortlist_size: int
    chunk_size: int
    kind_max_values: int
    kind_min_fill: float
    bloc_key_distinct: float
    sample_size: int
    header_score_floor: float
    header_margin: float
    drop_furniture: bool
    key_fill_floor: float
    key_distinct_floor: float
    instance_tolerance: float
    lexical_floor: float
    identifier_values_per_key: int
    nesting_floor: float
    stray_column_fill: float
    model_revision: str = ""
    model_identity: str = ""
    live_stages: tuple[str, ...] = STAGES
    binding_mode: str = "model"
    structure_mode: str = "model"
    refinement: str = "rules"
    grammar: bool = True
    accept_thresholds: dict[str, float] = field(default_factory=dict)
    work_dir: str = "work"
    disclosure: str = "local-model-may-see-values"
    lexicon_path: str | None = None
    serve_roots: tuple[str, ...] = ()
    limits: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | None) -> "Config":
        raw = tomllib.loads(
            resources.files("ftmap").joinpath("defaults.toml").read_text("utf-8")
        )
        if path is not None:
            with open(path, "rb") as fh:
                raw = _deep_merge(raw, tomllib.load(fh))
        url = raw["model"]["url"]
        if not _is_loopback(url):
            raise ValueError(
                f"model url {url!r} is not a loopback address. Cell values are "
                "sent to this endpoint; the boundary requires it to stay on "
                "this machine."
            )
        binding_mode = str(raw["plan"]["binding_mode"])
        if binding_mode not in BINDING_MODES:
            raise ValueError(
                f"[plan] binding_mode {binding_mode!r} is not one of "
                f"{list(BINDING_MODES)}")
        structure_mode = str(raw["plan"]["structure_mode"])
        if structure_mode not in STRUCTURE_MODES:
            raise ValueError(
                f"[plan] structure_mode {structure_mode!r} is not one of "
                f"{list(STRUCTURE_MODES)}")
        refinement = str(raw["plan"]["refinement"])
        if refinement not in REFINEMENT_MODES:
            raise ValueError(
                f"[plan] refinement {refinement!r} is not one of "
                f"{list(REFINEMENT_MODES)}")
        live_stages = tuple(str(x) for x in raw["model"]["live_stages"])
        unknown = [x for x in live_stages if x not in STAGES]
        if unknown:
            raise ValueError(
                f"[model] live_stages names {unknown}; the stages are "
                f"{list(STAGES)}")
        return cls(
            model_url=url,
            model_seed=int(raw["model"]["seed"]),
            model_temperature=float(raw["model"]["temperature"]),
            model_revision=str(raw["model"].get("revision") or ""),
            model_identity=str(raw["model"].get("identity") or ""),
            live_stages=live_stages,
            binding_mode=binding_mode,
            structure_mode=structure_mode,
            refinement=refinement,
            grammar=bool(raw["model"]["grammar"]),
            enable_thinking=bool(raw["model"]["enable_thinking"]),
            max_tokens=int(raw["model"]["max_tokens"]),
            shortlist_size=int(raw["plan"]["shortlist_size"]),
            chunk_size=int(raw["plan"]["chunk_size"]),
            kind_max_values=int(raw["plan"]["kind_max_values"]),
            kind_min_fill=float(raw["plan"]["kind_min_fill"]),
            bloc_key_distinct=float(raw["plan"]["bloc_key_distinct"]),
            sample_size=int(raw["plan"]["sample_size"]),
            lexicon_path=raw["plan"]["lexicon_path"] or None,
            lexical_floor=float(raw["plan"]["lexical_floor"]),
            identifier_values_per_key=int(raw["plan"]["identifier_values_per_key"]),
            nesting_floor=float(raw["plan"]["nesting_floor"]),
            stray_column_fill=float(raw["plan"]["stray_column_fill"]),
            header_score_floor=float(raw["layout"]["header_score_floor"]),
            header_margin=float(raw["layout"]["header_margin"]),
            drop_furniture=bool(raw["layout"]["drop_furniture"]),
            key_fill_floor=float(raw["validate"]["key_fill_floor"]),
            key_distinct_floor=float(raw["validate"]["key_distinct_floor"]),
            instance_tolerance=float(raw["etalon"]["instance_tolerance"]),
            accept_thresholds={
                k: float(v) for k, v in raw["validate"]["accept_thresholds"].items()
            },
            work_dir=raw["run"]["work_dir"],
            disclosure=raw["run"]["disclosure"],
            serve_roots=tuple(raw["serve"]["roots"]),
            limits={k: int(v) for k, v in raw["limits"].items()},
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "limits": dict(self.limits),
            "model": {
                "url": self.model_url,
                "revision": self.model_revision,
                "identity": self.model_identity,
                "seed": self.model_seed,
                "temperature": self.model_temperature,
                "enable_thinking": self.enable_thinking,
                "max_tokens": self.max_tokens,
                "live_stages": list(self.live_stages),
                "grammar": self.grammar,
            },
            "plan": {
                "binding_mode": self.binding_mode,
                "structure_mode": self.structure_mode,
                "refinement": self.refinement,
                "shortlist_size": self.shortlist_size,
                "chunk_size": self.chunk_size,
                "kind_max_values": self.kind_max_values,
                "kind_min_fill": self.kind_min_fill,
                "bloc_key_distinct": self.bloc_key_distinct,
                "sample_size": self.sample_size,
                "lexicon_path": self.lexicon_path,
                "lexical_floor": self.lexical_floor,
                "identifier_values_per_key": self.identifier_values_per_key,
                "nesting_floor": self.nesting_floor,
                "stray_column_fill": self.stray_column_fill,
            },
            "layout": {
                "header_score_floor": self.header_score_floor,
                "header_margin": self.header_margin,
                "drop_furniture": self.drop_furniture,
            },
            "validate": {
                "key_fill_floor": self.key_fill_floor,
                "key_distinct_floor": self.key_distinct_floor,
                "accept_thresholds": dict(self.accept_thresholds),
            },
            "etalon": {"instance_tolerance": self.instance_tolerance},
            "run": {"work_dir": self.work_dir, "disclosure": self.disclosure},
            "serve": {"roots": list(self.serve_roots)},
        }


@lru_cache(maxsize=1)
def packaged_config() -> "Config":
    """The PACKAGED defaults, parsed once per process."""
    return Config.load(None)


def packaged_limits() -> dict[str, int]:
    """The ceilings from the PACKAGED defaults, parsed once per process."""
    return packaged_config().limits
