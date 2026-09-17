import pytest
from ftmap.config import Config


def test_defaults_load_from_packaged_toml(tmp_path):
    cfg = Config.load(None)
    assert cfg.model_url == "http://127.0.0.1:8080"
    assert cfg.enable_thinking is False
    assert cfg.model_seed == 20260810
    assert cfg.accept_thresholds["date"] == 0.90
    assert cfg.accept_thresholds["name"] == 0.0


def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "ftmap.toml"
    p.write_text(
        "[model]\nurl = 'http://127.0.0.1:9999'\n"
        "[validate.accept_thresholds]\ndate = 0.5\n",
        encoding="utf-8",
    )
    cfg = Config.load(str(p))
    assert cfg.model_url == "http://127.0.0.1:9999"
    assert cfg.accept_thresholds["date"] == 0.5
    assert cfg.accept_thresholds["phone"] == 0.90  # untouched default survives


def test_non_loopback_model_url_is_rejected(tmp_path):
    p = tmp_path / "ftmap.toml"
    p.write_text("[model]\nurl = 'http://example.com:8080'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="loopback"):
        Config.load(str(p))


def test_manifest_round_trips_every_field():
    cfg = Config.load(None)
    m = cfg.as_manifest()
    assert m["model"]["seed"] == 20260810
    assert m["validate"]["accept_thresholds"]["identifier"] == 0.80


def test_serve_roots_default_to_nothing():
    """A server that has not been told what it may read may read nothing."""
    assert Config.load(None).serve_roots == ()


def test_serve_roots_come_from_the_toml(tmp_path):
    path = tmp_path / "ftmap.toml"
    path.write_text('[serve]\nroots = ["../data/files", "../article-1"]\n',
                    encoding="utf-8")
    cfg = Config.load(str(path))
    assert cfg.serve_roots == ("../data/files", "../article-1")


def test_serve_roots_are_recorded_in_the_manifest(tmp_path):
    path = tmp_path / "ftmap.toml"
    path.write_text('[serve]\nroots = ["../data/files"]\n', encoding="utf-8")
    assert Config.load(str(path)).as_manifest()["serve"] == {"roots": ["../data/files"]}
