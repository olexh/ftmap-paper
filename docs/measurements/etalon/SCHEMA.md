# The etalon format

An etalon is a hand-written answer for one source, made from the source alone
and before its run is looked at. `src/ftmap/etalon/` loads it, checks it and
scores a run against it. It answers four questions, scored separately:

| layer | question |
|---|---|
| `subject` | what schema is one row of this table? |
| `entities` | what entities does a row produce, and what is each keyed on? |
| `edges` | what connects to what? |
| `columns` | which property does each column carry? |

## The file

One YAML document per source, `<source>.etalon.yaml`:

```yaml
etalon: 2

source:
  path: corpus-external/ua_courts_register.csv
  sha256: 4a5789993d6d…      # the binding key; the path is for a reader
  sheet: ""

subject:
  answer: PublicBody
  accept: [Organization, LegalEntity]
  why: One row is one permanent state judicial body.

entities:
  - key: court
    schema: PublicBody
    accept: [Organization, LegalEntity]
    keys: [c0]
    why: 843 rows, 841 distinct names.

edges:
  - key: ownership
    kind: link                 # an edge schema with its own endpoints
    schema: Ownership
    accept: []                 # other edge schemata also right here
    source: owner
    target: airplane
  - key: authority_of
    kind: property             # an entity-typed property
    prop: Contract:authority
    source: contract
    target: authority

columns:
  - id: c0
    header: court_code
    role: key+property
    answer: PublicBody:registrationNumber
    accept: []
    accept_decline: false
    why: The join key, and a real identifier issued by a state register.
```

The source is matched on `sha256`, never on the path. Every key is required;
only its contents may be empty. `edges: []` means "no edge exists in this
file"; a file with no `edges:` key does not load.

`accept:` lists other qnames pre-registered as also right. `accept_decline:
true` says declining the column is also right.

## `role`

| role | meaning |
|---|---|
| `property` | carries a value that belongs on a property |
| `key` | individuates the row and nothing more; binding it is `STRETCHED` |
| `key+property` | identifies the row and is a fact about it; keying without emitting is `KEY-ONLY` |
| `unmappable` | real content no declared schema can hold |
| `not-data` | a row ordinal, an export artefact, a spacer |

## Edges

`kind: link` is a link entity (`Ownership`, `Membership`, `ContractAward`)
with its own source and target. `kind: property` is an entity-typed property
(`Contract:authority`); the pipeline cannot produce one, so it scores as
`EDGE-UNPRODUCIBLE` and stays out of the edge denominator.

## What the loader refuses

* A property the schema does not carry (`Vehicle:serialNumber`).
* An answer no declared entity could hold.
* A `key` or `key+property` column no entity keys on.
* An entity with no `keys:` field (`keys: []` is the answer for row ordinal).
* A relation FollowTheMoney does not declare.
* An edge endpoint out of range, or a `kind: property` edge whose `prop` is
  not entity-typed.
* A missing `sha256`.

## Verdicts

* **Columns**: `AGREE`, `BOTH-DEFENSIBLE`, `PIPELINE-WRONG`, `KEY-ONLY`,
  `MISSED`, `STRETCHED`. `AGREE` covers one property under two schema
  spellings when one schema descends from the other; `PublicBody:name` and
  `CourtCase:name` stay a disagreement.
* **Subject**: `SUBJECT-AGREE`, `SUBJECT-ACCEPTABLE`, `SUBJECT-WRONG`.
* **Entities**: `ENTITY-AGREE`, `ENTITY-GENERALISED` (an ancestor schema),
  `ENTITY-MISSING`, `ENTITY-EXTRA`. Pairing is by exact schema, then by
  generalisation, ties broken by overlapping key columns.
* **Keys**: `KEYS-AGREE`, `KEYS-EMPTY`, `KEYS-DIFFER`.
* **Edges**: `EDGE-AGREE`, `EDGE-ACCEPTABLE`, `EDGE-ENDPOINTS-WRONG`,
  `EDGE-MISSING`, `EDGE-EXTRA`, `EDGE-UNPRODUCIBLE`.

`ETALON-WRONG` is never issued: a comparison cannot find its reference at
fault.

## Running it

```bash
ftmap etalon docs/measurements/etalon --out work
```

A single `.etalon.yaml` may be given instead of a directory; `--json` prints
totals. The scorer reads a run's `summary.json` and nothing else, which is
the artefact the pipeline asserts is value-free.
