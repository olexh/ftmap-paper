"""One real exchange with the local server.

THE PROMPT SHAPE IS PART OF WHAT IS BEING TESTED. Measured on this machine:
the same column, the same three-value enum and the same grammar return
`unmapped` when the user message is a bare header-and-values line, and
`Person:birthDate` when it is shaped the way `binding_prompt` shapes it —
subject, declared entities, fill rate, generalized shape, detector verdict,
samples, and candidates carrying their ontology labels. The system prompt makes
no difference either way; the structure does.

So this test sends the production shape — and builds it with `binding_prompt`
itself rather than a pasted copy, because a pasted copy already drifted once:
when the roster grew its `identified by` line, this file kept sending the
retired single-line dialect and silently stopped measuring what production
sends. A smoke test built on a toy prompt would measure a prompt nobody sends,
and would fail for a reason that has nothing to do with the client.
"""

import dataclasses

import pytest

from ftmap.cli import _resolve
from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.plan.client import LlamaClient, ModelError
from ftmap.plan.prompt import BINDING_SYSTEM, binding_prompt
from ftmap.profile.columns import profile_frame
from ftmap.vocab.catalogue import Catalogue

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def cfg():
    """The configuration a REAL run would use, resolved the way a run resolves it.

    `Config.load(None)` reads the packaged defaults, where `[model] revision`
    is deliberately blank — so a live test built on it could never pass, and
    for four days did not: it failed on the missing revision before it ever
    opened a socket. The offline suite is right to use the defaults and this
    test is wrong to, because the thing under test is the deployment.

    So it resolves through `cli._resolve`, which is what `ftmap run` calls, and
    states the precondition as a precondition rather than as a stack trace.
    """
    cfg = Config.load(_resolve(None))
    if not cfg.model_revision:
        pytest.fail(
            "no deployment declared: write an `ftmap.toml` in the run "
            "directory with [model] revision and identity. The live gate "
            "measures a deployment, and an undeclared one is not a deployment. "
            "See docs/measurements/2026-08-25-live-model-contract.md.")
    return cfg

def _production_user() -> str:
    """One birth-date column, shaped by the production prompt builder."""
    offline = Config.load(None)
    frame = build_frame(
        Grid(rows=[["Дата народження"], ["17.09.1980"], ["01.02.1990"]],
             sheet="s", merges=[]),
        "/x.csv", "0" * 64, "sid/s", offline)
    profiles = profile_frame(frame, offline)
    column = profiles[0].id
    return binding_prompt(
        profiles,
        {column: ["person|Person:birthDate", "person|Person:name"]},
        "Person", {"person": "Person"}, Catalogue.load(),
        entities=[{"key": "person", "schema": "Person", "keys": [column]}])


USER = _production_user()


SCHEMA = {
    "type": "object",
    "properties": {"prop": {"type": "string",
                            "enum": ["Person:name", "Person:birthDate",
                                     "unmapped"]}},
    "required": ["prop"],
    "additionalProperties": False,
}


def test_real_server_honours_the_grammar_and_answers(cfg, tmp_path):
    with LlamaClient(cfg, cache_path=str(tmp_path / "c.jsonl")) as c:
        out = c.complete(BINDING_SYSTEM, USER, SCHEMA, max_tokens=120)
    assert out["prop"] == "Person:birthDate"


def test_a_live_answer_carries_the_producing_model_into_provenance(cfg, tmp_path):
    """What the manifest will say, checked against what actually answered.

    The identity check and the provenance it feeds were written against
    scripted fixtures, so until this ran, no real response's `model` field had
    ever been compared to anything. A fixture that agrees with the code that
    reads it agrees by construction.
    """
    with LlamaClient(cfg, cache_path=str(tmp_path / "c.jsonl")) as c:
        c.complete(BINDING_SYSTEM, USER, SCHEMA, max_tokens=120)
        provenance = c.model_provenance()

    assert provenance["revision"] == cfg.model_revision
    # Not `!= "unknown"`: the point is that the SERVER named itself and the
    # name reached provenance intact.
    assert provenance["producing_model"] == cfg.model_identity


def test_a_server_answering_under_another_name_fails_the_run(cfg, tmp_path):
    """The refusal, against a real response rather than a mocked one.

    A cache record claims a revision produced its bytes. That claim is only
    worth anything if a disagreeing answer is refused, and this is the only
    place the refusal meets an answer it did not construct.
    """
    wrong = dataclasses.replace(cfg, model_identity=cfg.model_identity + "-not")
    with LlamaClient(wrong, cache_path=str(tmp_path / "c.jsonl")) as c:
        with pytest.raises(ModelError) as caught:
            c.complete(BINDING_SYSTEM, USER, SCHEMA, max_tokens=120)

    assert cfg.model_identity in str(caught.value)
    # And nothing was written: an unverifiable answer must not become a cache
    # record that claims a revision.
    assert not (tmp_path / "c.jsonl").exists() or not (
        tmp_path / "c.jsonl").read_text(encoding="utf-8").strip()
