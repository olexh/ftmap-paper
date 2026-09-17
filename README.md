# ftmap

Source code, test suite, gold-standard etalons and measurement corpus for the
paper on compiling heterogeneous record tables into one FollowTheMoney entity
stream.

`ftmap` turns Ukrainian record tables (CSV, XLS, XLSX) into a FollowTheMoney
(FtM) entity stream with per-field provenance. A local llama.cpp model acts as
a compiler front-end: for each source it emits one mapping plan under a
grammar-constrained JSON schema; deterministic code validates the plan,
compiles it to a standard FtM mapping document, and FtM's own engine executes
it. Every run reproduces standalone with `ftm map --no-sign <file>.yml`,
without the model.

## Layout

    src/ftmap/            io -> profile -> plan -> build -> normalize
                          `plan/` is where the model is consulted; everything
                          from `build/` on is deterministic.
    src/ftmap/defaults.toml
                          Every number that can move a result. Echoed into
                          each run manifest.
    src/ftmap/vocab/      The FtM vocabulary shortlist and the Ukrainian label
                          catalogue.
    src/ftmap/etalon/     The etalon document format and the scorer.
    docs/measurements/etalon/
                          The gold standard: one `.etalon.yaml` per source.
                          Format in `SCHEMA.md`.
    corpus-external/      The measurement corpus: public open-data tables the
                          etalons are written against. Publisher, URL, licence
                          and retrieval date of each file are in `SOURCES.md`.
    tests/                The test suite. Fixtures are public open-data
                          exports (`tests/fixtures/README.md`);
                          `tests/golden/` is the expected corpus reading.
    tools/                The binding-baseline report the arm comparison in the
                          paper is computed with.

## Install

Python 3.13 or newer.

    uv sync --frozen

or, without `uv`:

    python -m venv .venv && .venv/bin/pip install -e . pytest jsonschema pytest-cov

## Tests

    uv run --frozen pytest -q             # offline suite
    uv run --frozen pytest -q -m corpus   # the etalons against the corpus files

Tests marked `live` need a running llama-server on loopback and `slow` are
resource benchmarks; both are deselected by default. See
`[tool.pytest.ini_options]` in `pyproject.toml`.

## Running the pipeline

    ftmap inventory corpus-external/person-graph              # what is in a corpus
    ftmap run corpus-external/person-graph --out work         # read, plan, validate, emit
    ftmap etalon docs/measurements/etalon --out work          # score the run

A model-backed run needs an `ftmap.toml` in the working directory declaring
the model deployment (`[model] revision` and `identity`); the model endpoint
must resolve to loopback, and a configuration pointing anywhere else is
refused.

## Boundary

Cell values from a source never leave the machine. The local model is the only
consumer of raw values, by construction. What may leave a run directory is
headers, generalized shapes, counts, rates and bounds; `tests/test_boundary.py`
enforces this, and `.gitignore` excludes run outputs by directory glob.
