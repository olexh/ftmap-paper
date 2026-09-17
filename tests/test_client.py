import dataclasses
import json
import os
import stat

import httpx
import pytest

from ftmap.config import Config
from ftmap.plan.client import _CACHE_FORMAT, LlamaClient, ModelError

# A SCRIPTED CLIENT DECLARES ITS OWN FIXTURE REVISION. The shipped default is
# blank on purpose — see `defaults.toml` — so `Config.load(None)` is exactly
# the unconfigured state a model-backed run must refuse. Every test that wants
# a working client says which deployment it is pretending to be, which is the
# same thing an operator has to do.
CFG = dataclasses.replace(Config.load(None), model_revision="fixture-rev-1")


def _cfg(**over) -> Config:
    return dataclasses.replace(CFG, **over)


def _transport(recorder, content='{"ok": true}'):
    def handler(request: httpx.Request) -> httpx.Response:
        # The client asks `/props` once at start; these transports are
        # not a server and say so.
        if request.method == "GET":
            return httpx.Response(404)
        recorder.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content, "reasoning_content": ""}}],
                  "usage": {"completion_tokens": 5}},
        )

    return httpx.MockTransport(handler)


def test_request_carries_the_settings_the_design_fixed(tmp_path):
    seen: list[dict] = []
    c = LlamaClient(CFG, cache_path=str(tmp_path / "c.jsonl"),
                    transport=_transport(seen))
    out = c.complete("sys", "user", {"type": "object"})
    assert out == {"ok": True}
    body = seen[0]
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["seed"] == 20260810
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["json_schema"]["strict"] is True


def test_the_servers_reported_model_is_captured(tmp_path):
    """I7: run.json records model_id so a published measurement is
    reproducible from its own manifest, even when what the server actually
    loaded differs from what ftmap.toml asked for."""
    def handler(request: httpx.Request) -> httpx.Response:
        # The client asks `/props` once at start; these transports are
        # not a server and say so.
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, json={
            "model": "gemma-4-26B-A4B-it",
            "choices": [{"message": {"content": '{"ok": true}', "reasoning_content": ""}}],
        })

    c = LlamaClient(CFG, cache_path=str(tmp_path / "c.jsonl"),
                    transport=httpx.MockTransport(handler))
    assert c.reported_model is None
    c.complete("sys", "user", {"type": "object"})
    assert c.reported_model == "gemma-4-26B-A4B-it"


def test_identical_prompt_is_served_from_cache(tmp_path):
    seen: list[dict] = []
    path = str(tmp_path / "c.jsonl")
    c = LlamaClient(CFG, cache_path=path, transport=_transport(seen))
    c.complete("sys", "user", {"type": "object"})
    c.complete("sys", "user", {"type": "object"})
    assert len(seen) == 1
    assert c.calls == 1 and c.cache_hits == 1

    fresh = LlamaClient(CFG, cache_path=path, transport=_transport(seen))
    fresh.complete("sys", "user", {"type": "object"})
    assert len(seen) == 1  # survives a new process


def test_unparseable_content_raises_rather_than_returning_junk(tmp_path):
    seen: list[dict] = []
    c = LlamaClient(CFG, cache_path=str(tmp_path / "c.jsonl"),
                    transport=_transport(seen, content="not json"))
    with pytest.raises(ModelError, match="did not return JSON"):
        c.complete("sys", "user", {"type": "object"})


def test_empty_content_names_the_thinking_channel(tmp_path):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "", "reasoning_content": "..."},
                         "finish_reason": "length"}],
            "usage": {"completion_tokens": 800},
        })

    c = LlamaClient(CFG, cache_path=str(tmp_path / "c.jsonl"),
                    transport=httpx.MockTransport(handler))
    with pytest.raises(ModelError, match="reasoning"):
        c.complete("sys", "user", {"type": "object"})


def test_cache_file_holds_real_prompt_values_so_it_is_written_owner_only(tmp_path):
    # The cache is a JSONL log of prompts, which carry real cell values from
    # `data/`. That is the boundary this test guards: whatever else changes,
    # the file may not be group- or world-readable.
    path = str(tmp_path / "c.jsonl")
    c = LlamaClient(CFG, cache_path=path, transport=_transport([]))
    c.complete("sys", "user", {"type": "object"})
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_a_truncated_last_line_is_skipped_not_fatal(tmp_path):
    # A run killed mid-write leaves a partial last JSONL line. Losing that one
    # cached answer should cost one model call on the next run, not crash
    # construction and refuse to start the whole run.
    path = str(tmp_path / "c.jsonl")
    seen: list[dict] = []
    c = LlamaClient(CFG, cache_path=path, transport=_transport(seen, content='{"n": 1}'))
    c.complete("sys", "one", {"type": "object"})
    c.complete("sys", "two", {"type": "object"})
    assert len(seen) == 2

    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 2
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(lines[0] + "\n")
        fh.write(lines[1][: len(lines[1]) // 2])  # simulate a kill mid-write

    fresh = LlamaClient(CFG, cache_path=path, transport=_transport(seen, content='{"n": 1}'))

    # The intact first entry is still served from cache.
    out_one = fresh.complete("sys", "one", {"type": "object"})
    assert out_one == {"n": 1}
    assert len(seen) == 2

    # The corrupted second entry was dropped, not crashed on, so it is a
    # cache miss and makes a fresh request.
    out_two = fresh.complete("sys", "two", {"type": "object"})
    assert out_two == {"n": 1}
    assert len(seen) == 3


def test_a_model_backed_call_without_a_declared_revision_fails_early(tmp_path):
    """Config.load must NOT enforce this, and `complete` must.

    Deterministic replay, `ftmap spec` and every non-model inspection load a
    Config; making the constructor demand a revision would force each of them
    to invent one it never uses. The boundary that cares is the one that puts
    an answer into a cache namespace and a manifest.
    """
    bare = Config.load(None)
    assert bare.model_revision == "", "the shipped default must be unset"

    client = LlamaClient(bare, cache_path=str(tmp_path / "c.jsonl"),
                         transport=_transport([]))
    with pytest.raises(ModelError) as excinfo:
        client.complete("sys", "user", {"type": "object"})
    assert "revision" in str(excinfo.value)
    client.close()


@pytest.mark.parametrize("field,value", [
    ("model_revision", "fixture-rev-2"),
    ("model_seed", 999),
    ("model_temperature", 0.7),
    ("enable_thinking", True),
    ("max_tokens", 4000),
])
def test_each_request_affecting_setting_changes_the_cache_key(tmp_path, field, value):
    """One parametrization per thing that was silently outside the old key.

    The old key was prompt and schema alone. Every setting below could change
    and the cache would answer with the previous deployment's output under the
    new configuration's name — an artefact stating it was produced by settings
    that never produced it.
    """
    path = str(tmp_path / "c.jsonl")
    seen: list[dict] = []
    with LlamaClient(CFG, cache_path=path, transport=_transport(seen)) as first:
        first.complete("sys", "user", {"type": "object"})
    assert len(seen) == 1

    with LlamaClient(_cfg(**{field: value}), cache_path=path,
                     transport=_transport(seen)) as second:
        second.complete("sys", "user", {"type": "object"})
        assert second.cache_hits == 0, f"{field} did not change the key"
    assert len(seen) == 2

    # ...and the unchanged request still hits, or the test above would pass
    # for a cache that simply never works.
    with LlamaClient(CFG, cache_path=path, transport=_transport(seen)) as third:
        third.complete("sys", "user", {"type": "object"})
        assert third.cache_hits == 1
    assert len(seen) == 2


def test_the_expected_model_identity_changes_the_cache_key(tmp_path):
    """Its own test, because it cannot share the others' silent transport.

    A configured identity makes every live answer verifiable, so a transport
    that reports no model is refused — correctly, and inconveniently for a
    parametrization whose point is the key rather than the check.
    """
    path = str(tmp_path / "c.jsonl")
    seen: list[dict] = []

    def transport(name):
        def handler(request: httpx.Request) -> httpx.Response:
            # The client asks `/props` once at start; these transports are
            # not a server and say so.
            if request.method == "GET":
                return httpx.Response(404)
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={
                "model": name,
                "choices": [{"message": {"content": '{"ok": true}',
                                         "reasoning_content": ""}}]})
        return httpx.MockTransport(handler)

    with LlamaClient(_cfg(model_identity="model-a"), cache_path=path,
                     transport=transport("model-a")) as first:
        first.complete("sys", "user", {"type": "object"})
    with LlamaClient(_cfg(model_identity="model-b"), cache_path=path,
                     transport=transport("model-b")) as second:
        second.complete("sys", "user", {"type": "object"})
        assert second.cache_hits == 0
    assert len(seen) == 2

    with LlamaClient(_cfg(model_identity="model-a"), cache_path=path,
                     transport=transport("model-a")) as third:
        third.complete("sys", "user", {"type": "object"})
        assert third.cache_hits == 1
    assert len(seen) == 2


def test_a_per_call_max_tokens_is_part_of_the_key(tmp_path):
    """`propose` varies this per call; the old key did not see it at all."""
    path = str(tmp_path / "c.jsonl")
    seen: list[dict] = []
    with LlamaClient(CFG, cache_path=path, transport=_transport(seen)) as c:
        c.complete("sys", "user", {"type": "object"}, max_tokens=100)
        c.complete("sys", "user", {"type": "object"}, max_tokens=200)
        assert c.cache_hits == 0
    assert len(seen) == 2


def test_a_cache_record_from_before_the_revision_existed_is_a_miss(tmp_path):
    """Not upgradable, because what it would need was never written down.

    A version-1 record holds prompt, schema and response and says nothing
    about which model or which settings produced them. Reusing one puts an
    answer of unknown origin into a manifest whose whole purpose is to say
    where every answer came from.
    """
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({
        "key": "whatever", "system": "sys", "user": "user",
        "response": {"stale": True}}) + "\n", encoding="utf-8")
    seen: list[dict] = []
    with LlamaClient(CFG, cache_path=str(path), transport=_transport(seen)) as c:
        out = c.complete("sys", "user", {"type": "object"})
    assert out == {"ok": True}
    assert len(seen) == 1, "a legacy record was reused"


def test_a_cache_hit_still_reports_which_model_produced_the_answer(tmp_path):
    """R11: an all-cache-hit run is still a model-backed run.

    `model_id: null` beside 148 planning decisions said the run had no model.
    It had one; it reused it.
    """
    path = str(tmp_path / "c.jsonl")

    def handler(request):
        return httpx.Response(200, json={
            "model": "qwen-fixture",
            "choices": [{"message": {"content": '{"ok": true}',
                                     "reasoning_content": ""}}]})

    with LlamaClient(CFG, cache_path=path,
                     transport=httpx.MockTransport(handler)) as live:
        live.complete("sys", "user", {"type": "object"})
        assert live.model_provenance() == {"revision": "fixture-rev-1",
                                           "producing_model": "qwen-fixture",
                                           "server": "not contacted"}

    with LlamaClient(CFG, cache_path=path, transport=_transport([])) as cached:
        cached.complete("sys", "user", {"type": "object"})
        assert cached.cache_hits == 1
        assert cached.model_provenance() == {"revision": "fixture-rev-1",
                                             "producing_model": "qwen-fixture",
                                             "server": "not contacted"}


@pytest.mark.parametrize("reported", ["something-else", None])
def test_a_live_answer_from_an_undeclared_model_fails_the_run(tmp_path, reported):
    """Including the unverifiable case.

    A server quietly loaded with a different build is the one way the declared
    revision is wrong in a manner that reaches every cached answer. A response
    naming no model at all cannot be checked, so it is refused for the same
    reason rather than trusted for a different one.
    """
    body = {"choices": [{"message": {"content": '{"ok": true}',
                                     "reasoning_content": ""}}]}
    if reported is not None:
        body["model"] = reported

    cache = tmp_path / "c.jsonl"
    with LlamaClient(_cfg(model_identity="declared-model"), cache_path=str(cache),
                     transport=httpx.MockTransport(
                         lambda r: httpx.Response(200, json=body))) as c:
        with pytest.raises(ModelError) as excinfo:
            c.complete("sys", "user", {"type": "object"})
    assert "declared-model" in str(excinfo.value)
    # And nothing was written, so a later run cannot pick it up as a hit.
    assert not cache.exists() or cache.read_text(encoding="utf-8") == ""


def test_a_matching_live_answer_is_accepted_and_records_its_producer(tmp_path):
    body = {"model": "declared-model",
            "choices": [{"message": {"content": '{"ok": true}',
                                     "reasoning_content": ""}}]}
    path = tmp_path / "c.jsonl"
    with LlamaClient(_cfg(model_identity="declared-model"), cache_path=str(path),
                     transport=httpx.MockTransport(
                         lambda r: httpx.Response(200, json=body))) as c:
        assert c.complete("sys", "user", {"type": "object"}) == {"ok": True}
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["v"] == _CACHE_FORMAT
    assert record["revision"] == "fixture-rev-1"
    assert record["model"] == "declared-model"


# --- 2026-09-03: the server that answers is part of the key -----------------

_PROPS = {"build_info": "b10360-48d22e295", "model_alias": "gemma-4-26B-A4B",
          "model_path": "/home/x/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
          "total_slots": 1, "default_generation_settings": {"n_ctx": 32768}}


def _server(recorder, props=_PROPS, content='{"ok": true}'):
    """A transport that answers `/props` like llama-server does, or 404 when
    `props` is None, and the chat endpoint like the others above."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            if props is None:
                return httpx.Response(404)
            return httpx.Response(200, json=props)
        recorder.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "gemma-4-26B-A4B",
            "choices": [{"message": {"content": content, "reasoning_content": ""}}]})
    return httpx.MockTransport(handler)


def test_the_server_that_answers_is_fingerprinted_into_the_key(tmp_path):
    """2026-09-01 §3: two cold runs on one server process agreed byte for
    byte, and both disagreed with 17 of 59 answers a cache from earlier
    processes held. The revision covers weights and build; `-c`, `--parallel`
    and the rest separate one process from the next and were outside the
    key. Now `/props` is."""
    path = str(tmp_path / "c.jsonl")
    seen: list[dict] = []
    with LlamaClient(CFG, cache_path=path, transport=_server(seen)) as c:
        c.complete("sys", "user", {"type": "object"})
        assert c.server and len(c.server) == 16
        assert c.model_provenance()["server"] == c.server
    other = dict(_PROPS, default_generation_settings={"n_ctx": 65536})
    with LlamaClient(CFG, cache_path=path, transport=_server(seen, props=other)) as c2:
        c2.complete("sys", "user", {"type": "object"})
        assert c2.server != c.server
    assert len(seen) == 2, "a different context size reused the other server's answer"


def test_a_record_written_under_another_server_is_a_miss(tmp_path):
    path = str(tmp_path / "c.jsonl")
    seen: list[dict] = []
    with LlamaClient(CFG, cache_path=path, transport=_server(seen)) as c:
        c.complete("sys", "user", {"type": "object"})
    with LlamaClient(CFG, cache_path=path, transport=_server(seen)) as again:
        again.complete("sys", "user", {"type": "object"})
        assert again.cache_hits == 1
    assert len(seen) == 1


def test_a_replay_with_no_server_reuses_the_cache_and_says_whose_it_was(tmp_path):
    """A rerun after a code change that touches no prompt is a legitimate
    run whose every answer is a hit, on a machine that may have no server up.
    It is served, and the manifest names the server whose answers it reused
    rather than inventing a fingerprint of its own."""
    path = str(tmp_path / "c.jsonl")
    seen: list[dict] = []
    with LlamaClient(CFG, cache_path=path, transport=_server(seen)) as c:
        c.complete("sys", "user", {"type": "object"})
        produced = c.server
    with LlamaClient(CFG, cache_path=path, transport=_server(seen, props=None)) as replay:
        assert replay.server is None
        assert replay.complete("sys", "user", {"type": "object"}) == {"ok": True}
        assert replay.cache_hits == 1
        assert replay.model_provenance()["server"] == produced
    assert len(seen) == 1
    with LlamaClient(CFG, cache_path=str(tmp_path / "empty.jsonl"),
                     transport=_server(seen, props=None)) as cold:
        assert cold.model_provenance()["server"] == "not contacted"


def test_a_version_two_record_is_a_miss(tmp_path):
    """It named no server, so nothing can say whether the process that wrote
    it would answer the same way today."""
    path = tmp_path / "c.jsonl"
    seen: list[dict] = []
    with LlamaClient(CFG, cache_path=str(path), transport=_server(seen)) as c:
        c.complete("sys", "user", {"type": "object"})
    rec = json.loads(path.read_text().strip())
    rec["v"] = 2
    del rec["server"]
    path.write_text(json.dumps(rec) + "\n")
    with LlamaClient(CFG, cache_path=str(path), transport=_server(seen)) as again:
        again.complete("sys", "user", {"type": "object"})
        assert again.cache_hits == 0
    assert len(seen) == 2
