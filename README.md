# ftmap

Code, tests, etalons and measurement corpus for the paper on compiling
heterogeneous record tables into a FollowTheMoney entity stream.

`ftmap` reads Ukrainian record tables (CSV, XLS, XLSX) and emits
FollowTheMoney (FtM) entities with per-field provenance. A local llama.cpp
model proposes one mapping plan per source under a grammar-constrained JSON
schema; deterministic code validates the plan, compiles it to a standard FtM
mapping document, and FtM's own engine executes it. Any run reproduces without
the model: `ftm map --no-sign <file>.yml`.

## Layout

| path | what |
|---|---|
| `src/ftmap/` | the pipeline: `io` → `profile` → `plan` → `build` → `normalize`. The model is consulted in `plan/`; everything from `build/` on is deterministic |
| `src/ftmap/defaults.toml` | every number that can move a result, echoed into each run manifest |
| `src/ftmap/vocab/` | the FtM vocabulary shortlist and the Ukrainian label catalogue |
| `src/ftmap/etalon/` | the etalon format and the scorer |
| `docs/measurements/etalon/` | the gold standard, one `.etalon.yaml` per source; format in `SCHEMA.md` |
| `corpus-external/` | the measurement corpus: 16 public open-data tables; provenance in `SOURCES.md` |
| `tests/` | the test suite; `fixtures/` are public open-data exports, `golden/` the expected corpus reading |
| `tools/` | the binding-baseline report behind the paper's arm comparison |

## Install

Python 3.13 or newer.

    uv sync --frozen

or

    python -m venv .venv && .venv/bin/pip install -e . pytest jsonschema pytest-cov

## Tests

    pytest -q             # offline suite
    pytest -q -m corpus   # the etalons against the corpus files

`live` tests need a llama-server on loopback and `slow` tests are resource
benchmarks; both are deselected by default.

## Run

    ftmap inventory corpus-external/person-graph
    ftmap run corpus-external/person-graph --out work
    ftmap etalon docs/measurements/etalon --out work

A model-backed run needs an `ftmap.toml` declaring the deployment
(`[model] revision` and `identity`). The model endpoint must be loopback;
anything else is refused.

## Boundary

Cell values never leave the machine. Only the local model sees them. What a
run directory may publish is headers, generalized shapes, counts, rates and
bounds; `tests/test_boundary.py` enforces this and `.gitignore` excludes run
outputs by directory glob.
