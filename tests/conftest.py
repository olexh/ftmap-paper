# tests/conftest.py
"""One real run, shared by the boundary tests.

Built by running the pipeline rather than by writing artefacts out by hand:
the tests' whole job is reading what the pipeline actually writes, and a
hand-made fixture would drift from that the first time a stage changed its
output. The model is scripted, so the run is deterministic and offline.
"""

import openpyxl
import pytest

from ftmap.config import Config
from ftmap.pipeline import run_corpus
from ftmap.plan.prompt import BINDING_MARKER
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()

# The one cell value the boundary tests search responses for.
SECRET_VALUE = "Коваленко Іван Петрович"


class ScriptedClient:
    """Binds c0 to Person:name and abstains on the rest."""

    calls = 0
    cache_hits = 0

    def complete(self, system, user, schema, max_tokens=None):
        if BINDING_MARKER not in user:
            return {"subject": "Person",
                    "entities": [{"key": "person", "schema": "Person",
                                  "keys": ["c0"]}],
                    "edges": []}
        out = []
        for line in user.splitlines():
            if line.startswith("c") and ":" in line:
                cid = line.split(":", 1)[0].strip()
                binding = "person|Person:name" if cid == "c0" else "unmapped"
                out.append({"column": cid, "binding": binding, "why": "x"})
        return {"bindings": out}


@pytest.fixture
def secret_value() -> str:
    """The one cell value the boundary tests search responses for.

    A fixture rather than an import, because `tests/` is not a package and
    `from tests.conftest import ...` depends on how pytest was invoked.
    """
    return SECRET_VALUE


@pytest.fixture
def api_out(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    wb = openpyxl.Workbook()
    wb.active.title = "арк"
    wb.active.append(["ПІБ", "Дата народження"])
    wb.active.append([SECRET_VALUE, "17.09.1980"])
    wb.save(corpus / "a.xlsx")
    out = str(tmp_path / "work")
    run_corpus(str(corpus), CFG, CAT, ScriptedClient(), out)
    return out
