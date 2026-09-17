# tests/test_canonical.py
from ftmap.normalize.canonical import (acceptance, canonicalize, is_sentinel,
                                       column_evidence)


def test_ukrainian_and_russian_written_dates():
    assert canonicalize("17.09.1980", "date").value == "1980-09-17"
    assert canonicalize("1980-09-17", "date").value == "1980-09-17"
    assert canonicalize("17/09/1980", "date").value == "1980-09-17"
    # A bare year is a date where the column says so, and nowhere else — see
    # `test_a_bare_number_is_a_date_only_where_the_column_corroborates_it`.
    assert canonicalize("1980", "date", column_evidence(["1980"])).value == "1980"
    r = canonicalize("не вказано", "date")
    assert r.ok is False and "date" in r.reason


def test_two_digit_years_use_a_stated_century_rule():
    assert canonicalize("17.09.80", "date").value == "1980-09-17"
    assert canonicalize("17.09.05", "date").value == "2005-09-17"


def test_the_century_pivot_never_touches_a_four_digit_year():
    """`y < 100` also caught a genuinely four-digit year that happens to be
    small: "0000-01-01" parsed to y=0, which is under 100, so the pivot
    silently turned a placeholder into "2000-01-01" — a plausible date
    manufactured from a value that should have been reported uncanonicalizable.
    Gated on the matched text's length instead."""
    r = canonicalize("0000-01-01", "date")
    assert r.ok is False
    # A genuine early four-digit year must still canonicalize untouched.
    assert canonicalize("0099-01-01", "date").value == "0099-01-01"


def test_iso_8601_datetimes_canonicalize_and_keep_their_clock():
    """Measured on ua_war_sanctions (OpenSanctions bulk export), whose
    `first_seen`/`last_seen`/`last_change` are `2026-08-19T05:41:01`: the date
    pack had no clock-bearing row, so acceptance was 0.00, the rule layer
    rejected shape `dddd-dd-ddLdd:dd:dd`, and the repair round unmapped three
    bindings that were correct.

    The timestamp is PRESERVED, not truncated: FtM date properties take a full
    ISO timestamp, and `retrievedAt`/`createdAt` are about the moment. A
    numeric offset is moved onto UTC because FtM's date type is naive and has
    nowhere to keep the zone — otherwise the same instant written two ways
    canonicalizes to two values.
    """
    assert canonicalize("2026-08-19T05:41:01", "date").value == "2026-08-19T05:41:01"
    assert canonicalize("2026-08-19T05:41:01Z", "date").value == "2026-08-19T05:41:01"
    assert canonicalize("2026-08-19T05:41:01+02:00", "date").value == "2026-08-19T03:41:01"
    # Optional pieces the same exporters vary on.
    assert canonicalize("2026-08-19T05:41", "date").value == "2026-08-19T05:41:00"
    assert canonicalize("2026-08-19T05:41:01.123456Z", "date").value == "2026-08-19T05:41:01"
    # A broken clock is still reported, not repaired into something plausible.
    assert canonicalize("2026-08-19T25:41:01", "date").ok is False
    # The Y-D-M swap is date-only: ISO fixes the field order, so a month above
    # twelve beside a clock is a broken value, not a transposed one.
    assert canonicalize("2026-19-08T05:41:01", "date").ok is False
    # And the detector agrees, so such a column still retrieves date candidates.
    from ftmap.profile.detectors import is_date
    assert is_date("2026-08-19T05:41:01Z") is True


def test_excel_serial_dates_survive():
    """Spreadsheets genuinely store dates as a day count, so a column of them
    must still reach a date. What changed is the evidence required, not the
    form: a serial is admitted where the column's other numbers are serials
    too, which is what a real serial-date column looks like."""
    serials = ["29481", "29482", "30000", "44927"]
    got = canonicalize("29481", "date", column_evidence(serials))
    assert got.value == "1980-09-17"
    assert acceptance(serials, "date")[0] == 1.0
    # And alone, with no column behind it, it is a five-digit number.
    assert canonicalize("29481", "date").ok is False


def test_a_bare_number_is_a_date_only_where_the_column_corroborates_it():
    """Measured on the МВС vehicle registry (39 607 rows). CAPACITY is engine
    displacement in cc and TOTAL_WEIGHT/OWN_WEIGHT are kerb weight in kg, and
    all three canonicalized partly as `date` — 0.27, 0.28 and 0.10 — because
    `1956` cc and `1998` kg fall inside a plausible year range while `2993` cc
    does not. Partial acceptance is the dangerous shape: the column half-passes
    instead of failing, and the acceptance floor sees a middling number rather
    than a clear no.

    MAKE_YEAR is the honest case in the same file — it IS a year, and
    `Vehicle:buildDate` is a real target for it — so the fix cannot be "a bare
    year is never a date". The column is what separates them.
    """
    displacement = ["1956", "2993", "1598", "1248", "1968", "2000", "1390",
                    "3471", "11946", "998"]
    weight = ["1922", "2575", "1505", "1945", "990", "3500", "2100", "1240"]
    years = ["1949", "1956", "2015", "2013", "2026", "1980", "2001"]

    for column in (displacement, weight):
        share, _ = acceptance(column, "date")
        assert share == 0.0, column
        for v in column:
            assert canonicalize(v, "date", column_evidence(column)).ok is False

    # The year column keeps every value, including the two that also read as a
    # displacement and a weight.
    assert acceptance(years, "date")[0] == 1.0
    assert canonicalize("1956", "date", column_evidence(years)).value == "1956"


def test_a_column_says_nothing_about_the_written_date_forms():
    """Corroboration gates the two BARE-NUMBER forms and nothing else. A
    written date announces itself with its own punctuation, so a column of
    junk cannot take it away and a column of years cannot repair it."""
    junk = ["не вказано", "-", "0", "17.09.1980"]
    assert canonicalize("17.09.1980", "date", column_evidence(junk)).value == "1980-09-17"
    assert canonicalize("2026-08-19T05:41:01", "date",
                        column_evidence(junk)).value == "2026-08-19T05:41:01"
    assert canonicalize("17.09.80", "date", column_evidence(junk)).value == "1980-09-17"
    # And a year column does not make a broken written date canonicalize.
    years = ["1949", "2015", "2026"]
    assert canonicalize("2026-19-08T05:41:01", "date", column_evidence(years)).ok is False
    assert canonicalize("0000-01-01", "date", column_evidence(years)).ok is False


def test_the_refusal_of_a_bare_number_says_it_was_the_column():
    """No silent drops: a value refused for want of corroboration must be
    distinguishable in `rejects.jsonl` from one no date form matches at all."""
    r = canonicalize("1956", "date")
    assert r.ok is False and r.canonicalizer == "date"
    assert "column" in r.reason
    assert "column" not in canonicalize("не вказано", "date").reason


def test_column_evidence_scores_every_bare_number_against_one_denominator():
    """CAPACITY holds 895 five-digit displacements (10 000-19 998 cc), every
    one of them inside the Excel serial window. Scored as their own
    population they corroborate each other perfectly; scored against the whole
    column's 35 989 numbers they are 0.025 of it. One denominator is what
    keeps a measurement column from smuggling part of itself through."""
    mixed = ["11946", "12000", "1598", "2993", "998", "450", "1248"]
    assert column_evidence(mixed).numeric_dates is False
    assert canonicalize("11946", "date", column_evidence(mixed)).ok is False
    # A column that really is serials has no such minority to hide.
    assert column_evidence(["11946", "12000", "29481"]).numeric_dates is True


def test_a_column_with_no_bare_numbers_admits_none():
    assert column_evidence([]).numeric_dates is False
    assert column_evidence(["17.09.1980", None, "  "]).numeric_dates is False


def test_a_year_recorded_beside_full_dates_still_canonicalizes():
    """A register that knows only the year for some records writes it bare
    beside the full dates. The bare numbers in such a column are all plausible
    years, so they are admitted — this is the case the corroboration test must
    not cost."""
    column = ["17.09.1980", "1975", "01.02.1990", "1982", "1949"]
    share, rejected = acceptance(column, "date")
    assert share == 1.0 and rejected == []


def test_phones_of_both_countries_reach_e164():
    assert canonicalize("0501234567", "phone").value == "+380501234567"
    assert canonicalize("(050) 123-45-67", "phone").value == "+380501234567"
    assert canonicalize("+380501234567", "phone").value == "+380501234567"
    assert canonicalize("79253902086", "phone").value == "+79253902086"
    assert canonicalize("8 925 390 20 86", "phone").value == "+79253902086"
    assert canonicalize("не має", "phone").ok is False


def test_gender_codes_of_this_corpus():
    assert canonicalize("ч", "gender").value == "male"
    assert canonicalize("2", "gender").value == "female"
    assert canonicalize("Іван", "gender").ok is False


def test_names_are_tidied_but_never_transliterated_or_reordered():
    r = canonicalize("  Коваленко   Іван  Петрович ", "name")
    assert r.value == "Коваленко Іван Петрович"


def test_identifiers_lose_separators_and_keep_leading_zeroes():
    assert canonicalize("05480654", "identifier").value == "05480654"
    assert canonicalize("591-903-748.10", "identifier").value == "59190374810"


def test_a_document_number_containing_a_slash_stays_one_value():
    """«12345/67» is one document number, and `_canon_identifier` strips the
    slash as internal punctuation. Splitting on it produced two values that
    both passed, so the every-part guard could not catch it."""
    got = canonicalize("12345/67", "identifier")
    assert got.ok is True
    assert got.value == "1234567"


def test_an_identifier_gate_that_accepts_prose_is_not_a_gate():
    """Measured on deputies/7f79ded2: a biography, a surname, a party name and
    a street name all scored 1.000 as `identifier` against a 0.80 floor, so the
    threshold could never fire. With one token and one digit required they
    score 0.000, while the orgbook ЄДРПОУ column holds at 0.980."""
    for prose in ("Громадянин України, народився 25.05.1970",
                  "Баличев", "вул. Центральна",
                  "Одеська регіональна організація"):
        assert canonicalize(prose, "identifier").ok is False, prose
    assert canonicalize("22819278", "identifier").ok is True


def test_a_url_cell_is_split_without_cutting_the_scheme():
    """`/` separates two phone numbers and is part of a URL's own syntax, so
    the separator set is per type."""
    got = canonicalize("https://a.example/page, https://b.example/page", "url")
    assert got.ok is True
    assert got.value == "https://a.example/page;https://b.example/page"


def test_country_names_the_registers_print():
    assert canonicalize("Україна", "country").value == "ua"
    assert canonicalize("Росія", "country").value == "ru"
    assert canonicalize("Республіка Білорусь", "country").value == "by"
    assert canonicalize("UA", "country").value == "ua"


def test_the_long_official_form_is_rejected_with_a_reason_not_guessed():
    """Measured: neither rigour nor FtM resolves this, and it is exactly what
    the official Ukrainian registers publish. Reporting a rejection is honest;
    inventing a country code would not be."""
    r = canonicalize("Сполучене Королівство Великої Британії та Північної Ірландії",
                     "country")
    assert r.ok is False
    assert "territory" in r.reason


def test_email_and_url_use_ftm_itself():
    # FtM's EmailType.clean_text lowercases only the domain part and preserves
    # the local part's case (RFC 5321 technically treats it as case-sensitive).
    # Verified by reading followthemoney 4.10.1's installed source: this is
    # what "use FtM's own clean()" actually produces, not full lowercasing.
    assert canonicalize("Ivan@Example.COM", "email").value == "Ivan@example.com"
    assert canonicalize("не пошта", "email").ok is False


def test_acceptance_asks_a_declared_format_of_the_canonical_value():
    """A property's `format` narrows its type, and the type's canonicalizer
    knows nothing about it: these OpenSanctions ids are single tokens carrying
    a digit, so `identifier` takes all four, and followthemoney's `wikidata`
    check takes none. Measured on ua_war_sanctions.targets.simple.csv, that
    gap cost 5042 of 5621 values on one column, refused one at a time inside
    the engine after the gate had accepted the binding.

    ASKED OF THE CANONICAL VALUE. `_canon_identifier` strips the internal
    punctuation, so `NK-228j...` reaches the engine as `NK228j...`, and a
    format check run on the cell would be judging a string the engine never
    sees. The REJECTED list is still the cells, because that list is what the
    repair prompt and the rejection shapes are built from.
    """
    os_ids = ["NK-228jBYSTdUSvbZvsKsiHh6", "NK-23p2d4vMT5sJtQ845GyzJt",
              "NK-24ZomtS59XsEEB94Qcqy7S", "NK-26WkuEefVYLmYk8aTHMC9E"]
    assert acceptance(os_ids, "identifier") == (1.0, [])
    share, rejected = acceptance(os_ids, "identifier", "wikidata")
    assert share == 0.0
    assert rejected == os_ids
    # The same property on a column that can satisfy it. `q7259` is accepted
    # because followthemoney's own format normalizes it to `Q7259`.
    assert acceptance(["Q6319", "Q42", "q7259"], "identifier", "wikidata") == (1.0, [])


def test_a_packed_cell_must_clear_the_format_in_every_part():
    """The canonical form of a packed cell is its parts joined on `;`, and the
    engine is told to split there, so one bad part is one refused value —
    exactly what the engine will do with it."""
    assert acceptance(["Q42;Q6319"], "identifier", "wikidata") == (1.0, [])
    share, rejected = acceptance(["Q42;NK228jBYSTdUSvbZvsKsiHh6"],
                                 "identifier", "wikidata")
    assert share == 0.0
    assert rejected == ["Q42;NK228jBYSTdUSvbZvsKsiHh6"]


def test_a_format_followthemoney_does_not_know_refuses_nothing():
    """`clean_text` ignores an unregistered format string, and so does this:
    a format this followthemoney has never heard of must not silently reject
    a column the engine would accept whole."""
    assert acceptance(["12345", "АА-77"], "identifier", "no-such-format") == (1.0, [])


def test_acceptance_reports_the_share_and_the_rejects():
    share, rejected = acceptance(["17.09.1980", "01.02.1990", "12 січня", "-"],
                                 "date")
    assert share == 2 / 3
    assert set(rejected) == {"12 січня"}


def test_a_placeholder_is_not_a_value_that_failed():
    """`-`, `не вказано` and `XXX` are how a publisher writes "empty", and
    counting them as failures charges a binding for the spelling. Measured on
    the ICIJ entities file: `XXX` and `Undetermined` on 14 % of the rows took
    `Company:jurisdiction` to 0.86 against a 0.90 floor and the etalon's own
    answer was refused. What a column holds instead of values is reported by
    its fill rate and by `sentinel_share`, which reads the same list."""
    share, rejected = acceptance(["17.09.1980", "-", "не вказано"], "date")
    assert share == 1.0 and rejected == []


def test_one_cell_holding_two_phone_numbers():
    """Measured on deputies/7f79ded2: a quarter of its telephone column holds
    two real office numbers in one cell, and judged one value at a time the
    column scored 0.75 and the gate discarded it whole."""
    for raw in ("380486852018, 380486864319", "380486852018 380486864319",
                "380486852018;380486864319"):
        got = canonicalize(raw, "phone")
        assert got.ok is True, raw
        assert got.value == "+380486852018;+380486864319"
        assert got.canonicalizer == "phonex2"


def test_splitting_is_refused_when_a_part_is_not_a_value_of_that_type():
    """The guard that makes splitting on a bare space safe."""
    got = canonicalize("Коваленко Іван", "phone")
    assert got.ok is False
    assert canonicalize("0501234567 не телефон", "phone").ok is False


def test_free_text_is_never_split():
    """Splitting a note on a comma would be vandalism."""
    got = canonicalize("Має вищу освіту, працює у раді", "text")
    assert got.ok is True
    assert got.value == "Має вищу освіту, працює у раді"


def test_a_middle_component_over_twelve_cannot_be_a_month():
    """Measured on deputies/9e4f7a6d, which publishes birth dates as Y-D-M.
    Without the swap the column scored 0.265 and was discarded whole."""
    assert canonicalize("1978-28-07", "date").value == "1978-07-28"
    assert canonicalize("1988-30-12", "date").value == "1988-12-30"
    # Genuinely ambiguous: nothing in one value can settle it, so Y-M-D stands.
    assert canonicalize("1960-07-09", "date").value == "1960-07-09"


def test_multivalue_splitting_also_reaches_ftm_backed_types():
    """`email` and `url` are recognised through `_canon_ftm` (the
    FTM_SAFE_TYPES branch), not through CANONICALIZERS like `phone` and
    `identifier`. An earlier draft's `canonicalize()` only ever consulted
    `_canon_multivalue` from the CANONICALIZERS branch, so a cell holding two
    real emails would never actually take the multi-value path despite
    `email` being declared a member of MULTIVALUE_TYPES. Not one of the six
    tests specified for this change exercises this branch, so it is covered
    here rather than left to be caught by inspection alone next time."""
    got = canonicalize("Ivan@Example.com, Petro@Example.com", "email")
    assert got.ok is True
    assert got.value == "Ivan@example.com;Petro@example.com"
    assert got.canonicalizer == "emailx2"
    # The guard still applies: one real address plus noise is refused, not
    # silently reduced to the address alone.
    assert canonicalize("Ivan@Example.com не має", "email").ok is False


def test_a_jurisdiction_is_added_as_data_not_as_a_branch():
    """The point of the locale pack: a new country is a row in a table."""
    from ftmap.normalize import canonical

    assert canonicalize("+491701234567", "phone").ok is False
    original = canonical.PHONE_RULES
    canonical.PHONE_RULES = original + (
        {"name": "de", "cc": "49", "nsn": 10, "trunk": "0", "nsn_prefixes": ()},
    )
    try:
        assert canonicalize("+491701234567", "phone").value == "+491701234567"
    finally:
        canonical.PHONE_RULES = original


def test_a_caller_can_register_its_own_canonicalizer():
    from ftmap.normalize.canonical import CanonResult, canonicalize, register

    original = dict(__import__("ftmap.normalize.canonical", fromlist=["x"]).CANONICALIZERS)
    register("checksum", lambda v: CanonResult(True, v.upper(), None, "shouty"))
    try:
        got = canonicalize("abc", "checksum")
        assert got.value == "ABC" and got.canonicalizer == "shouty"
    finally:
        import ftmap.normalize.canonical as c
        c.CANONICALIZERS.clear()
        c.CANONICALIZERS.update(original)


def test_types_with_no_information_accept_everything():
    assert canonicalize("anything at all", "string").ok is True
    assert acceptance(["x", "y"], "text")[0] == 1.0


def test_a_registered_canonicalizer_reaches_the_gate_too():
    """`register()` is the seam for knowledge this package does not have. If
    acceptance short-circuits on FREE_TYPES before consulting the registry,
    the seam reaches canonicalize() but not the gate, which is where it
    matters. `checksum` is in FREE_TYPES and is exactly what a caller would
    register a real validator for."""
    import ftmap.normalize.canonical as c
    from ftmap.normalize.canonical import CanonResult

    original = dict(c.CANONICALIZERS)
    c.register("checksum", lambda v: CanonResult(False, None, "bad", "strict"))
    try:
        assert canonicalize("ANY", "checksum").ok is False
        share, rejected = acceptance(["ANY", "OTHER"], "checksum")
        assert share == 0.0
        assert rejected == ["ANY", "OTHER"]
    finally:
        c.CANONICALIZERS.clear()
        c.CANONICALIZERS.update(original)


def test_an_empty_column_cannot_contradict_a_binding():
    """No values is no evidence against, not evidence of rejection. Task 14
    thresholds this number, and 0.0 would make a blank column indistinguishable
    from a column whose every value failed."""
    assert acceptance([], "date") == (1.0, [])
    assert acceptance(["", None, "   "], "date") == (1.0, [])


# ---------------------------------------------------------------------------
# The multivalue policy table. Three real packed cells, and the three real
# unpacked ones that a careless separator would have destroyed. All six values
# are verbatim from `corpus-external/ua_war_sanctions.targets.simple.csv`.
# ---------------------------------------------------------------------------


def test_packed_aliases_split_into_one_value_each_in_both_scripts():
    """Measured on ua_war_sanctions (OpenSanctions bulk export): the `aliases`
    column filled 4007 cells, 2330 of them holding several aliases packed with
    `;`. Because `name` was in neither the multivalue set nor the separator
    table, every one of those reached `statements.csv` as ONE `alias` value —
    and followthemoney marks `alias` matchable, so a composite went into record
    linkage as a single token that can never match anything. That is worse than
    dropping it: it looks like data.

    `name` is a FREE_TYPE, so the every-part guard is vacuous here — nothing
    downstream of the separator can refuse this split. The separator itself is
    the guarantee, which is why it is `;` and nothing else."""
    got = canonicalize("ІВАХА Денис Володимирович;ИВАХА Денис Владимирович", "name")
    assert got.ok is True
    assert got.canonicalizer == "namex2"
    assert got.value.split(";") == ["ІВАХА Денис Володимирович",
                                    "ИВАХА Денис Владимирович"]

    got = canonicalize("Ceram Sea;Gold Sun;Nurkez", "name")
    assert got.ok is True
    assert got.canonicalizer == "namex3"
    assert got.value.split(";") == ["Ceram Sea", "Gold Sun", "Nurkez"]


def test_a_name_is_never_split_on_a_comma_or_a_slash():
    """The risk the policy has to address, in person. In the same 5621-row file
    the `name` column holds `;` in 0 cells but `,` in 36 and `/` in 317, and
    both are content: «SUN SCIENCE INTERNATIONAL CO., LIMITED» is one company,
    and «... c/o PJSC Sakhalin Shipping Company» is one name.

    «ABDULLIN Rinat Kamilievich / KingR» genuinely IS two names packed with a
    slash, and it stays unsplit anyway. That is the deliberate cost: for a type
    whose validator accepts anything there is no way to tell that cell apart
    from `c/o`, so the ambiguous separator is refused for both and the loss is
    one visible unsplit cell rather than 317 shredded names."""
    for one in ("SUN SCIENCE INTERNATIONAL CO., LIMITED",
                "Vostok Trade Invest JSC c/o PJSC Sakhalin Shipping Company",
                "ABDULLIN Rinat Kamilievich / KingR"):
        got = canonicalize(one, "name")
        assert got.ok is True, one
        assert got.value == one
        assert got.canonicalizer == "whitespace", one


def test_an_address_splits_on_a_semicolon_but_never_on_its_own_commas():
    """Every one of the 1436 filled `addresses` cells contains a comma and 148
    contain a slash — both are address syntax — while 31 contain `;`, and each
    of those is two spellings of one address."""
    packed = ("105318, Moscow, Tkatskaya Street, 19, floor 4, room 400;"
              "105318, Moscow, st. Tkatskaya, 19, room 4/400")
    got = canonicalize(packed, "address")
    assert got.ok is True
    assert got.canonicalizer == "addressx2"
    assert got.value.split(";") == [
        "105318, Moscow, Tkatskaya Street, 19, floor 4, room 400",
        "105318, Moscow, st. Tkatskaya, 19, room 4/400",
    ]
    one = "390000, Ryazan region, Ryazan, Seminarskaya Street, 32"
    assert canonicalize(one, "address").value == one


def test_a_packed_country_cell_validates_per_code_not_as_a_string():
    """The same packing rejected the `countries` column outright: the rule
    layer said "acceptance 0.74 below 0.90 for type country; rejected shapes
    LL;LL, LL;LL;LL" because it validated `lr;pa;sg;st` rather than its parts.
    Measured over that column's 5090 filled cells, acceptance is 0.7395 judged
    whole and 1.0000 judged per code."""
    got = canonicalize("lr;pa;sg;st", "country")
    assert got.ok is True
    assert got.canonicalizer == "countryx4"
    assert got.value == "lr;pa;sg;st"
    assert canonicalize("ru;ua", "country").value == "ru;ua"
    # The gate is what actually consumed the packed string.
    share, _ = acceptance(["ru", "ru;ua", "lr;pa;sg;st", "gm;gr;pa"], "country")
    assert share == 1.0


def test_a_country_is_not_split_on_the_spaces_inside_its_own_name():
    """A long-form country name is several words, so a whitespace separator
    would leave every one of them resting on the every-part guard alone. The
    pattern is delimiters only, so the guard is a second line of defence."""
    assert canonicalize("Республіка Білорусь", "country").value == "by"
    assert canonicalize("United Arab Emirates", "country").value == "ae"
    # A cell that is a country name plus noise is still rejected, not reduced.
    assert canonicalize("Росія та інші", "country").ok is False


def test_packed_identifiers_still_split_and_a_slash_is_still_internal():
    """The half that already worked, kept working: 5407 filled `identifiers`
    cells produced 9274 `idNumber` statements, none containing `;`. `/` stays
    internal punctuation for this type — «12345/67» is one document number, and
    splitting on it would turn one value into two that both pass, which the
    every-part guard cannot catch."""
    got = canonicalize("1025400524313;5401103595", "identifier")
    assert got.ok is True
    assert got.canonicalizer == "identifierx2"
    assert got.value == "1025400524313;5401103595"
    assert canonicalize("76403757;IMO6495856", "identifier").value == \
        "76403757;IMO6495856"
    assert canonicalize("12345/67", "identifier").value == "1234567"


def test_two_packed_dates_in_one_cell():
    """One `birth_date` cell in the same file holds `1952-08-20;2002-05-07`.
    `date` separates on `;` alone: `/` is the date's own field separator, so
    «17/09/1980» must stay one date."""
    got = canonicalize("1952-08-20;2002-05-07", "date")
    assert got.ok is True
    assert got.canonicalizer == "datex2"
    assert got.value == "1952-08-20;2002-05-07"
    assert canonicalize("17/09/1980", "date").value == "1980-09-17"


def test_a_free_type_split_still_has_to_produce_values_not_debris():
    """EVERY_PART is vacuous for a FREE_TYPE — `_canon_free` accepts
    everything — so the one thing still checkable without a validator is that
    each part holds a letter or a digit. A trailing or doubled separator
    otherwise manufactures an empty alias."""
    assert canonicalize("Ivan; ; Petro", "name").ok is True
    assert canonicalize("Ivan; ; Petro", "name").value == "Ivan;Petro"
    # A part that is neither empty nor a value costs ITSELF and not the two
    # aliases beside it: `;` is this type's measured packer, so the cell is a
    # real pack and «-» is a bad value inside one. It used to fall through to
    # `_canon_free`, which returned «Ivan; - ; Petro» unchanged — a `;` inside
    # a value `_binding_source` has told the engine to split, so the engine
    # made three names out of it, the middle one being this debris.
    got = canonicalize("Ivan; - ; Petro", "name")
    assert got.ok is True
    assert got.value == "Ivan;Petro"
    assert got.canonicalizer == "namex2of3"
    assert [part for part, _ in got.refused] == ["-"]


def test_membership_and_separators_cannot_drift_apart():
    """The shape of the original bug. `MULTIVALUE_TYPES` was a hand-kept set
    BESIDE a separator table: `identifier` was in both and split, `name` and
    `country` were in neither and did not. Deriving the set from the policy
    table makes that state unrepresentable, and every consumer
    (`plan.compile._binding_source`, `build.execute._offered_raw`,
    `build.emit.coverage`) reads the derived name."""
    from ftmap.normalize.canonical import MULTIVALUE_POLICY, MULTIVALUE_TYPES

    assert MULTIVALUE_TYPES == frozenset(MULTIVALUE_POLICY)
    assert {"name", "country", "address", "identifier"} <= MULTIVALUE_TYPES
    # A free-text note is in neither, and splitting one would be vandalism.
    assert "text" not in MULTIVALUE_TYPES and "string" not in MULTIVALUE_TYPES


def test_a_new_multivalued_type_is_a_row_in_the_table_not_a_branch():
    """The same test the locale pack gets: adding a type must not need an edit
    anywhere else. `topic` is FtM-backed and not in the table."""
    import ftmap.normalize.canonical as c

    assert canonicalize("role.pep;sanction", "topic").canonicalizer != "topicx2"
    original = dict(c.MULTIVALUE_POLICY)
    c.MULTIVALUE_POLICY["topic"] = (c.re.compile(r"\s*;\s*"), c.EVERY_PART)
    try:
        got = canonicalize("role.pep;sanction", "topic")
        assert got.ok is True and got.canonicalizer == "topicx2"
        # AND THE MEMBERSHIP NAME AGREES. `MULTIVALUE_TYPES` is a live view of
        # the table, so `_binding_source` tells the engine to split this column
        # too. A frozenset snapshot would have split the value here and left
        # the compiled mapping reading it whole.
        assert "topic" in c.MULTIVALUE_TYPES
    finally:
        c.MULTIVALUE_POLICY.clear()
        c.MULTIVALUE_POLICY.update(original)


# One clean value per multi-valued type, so the tests below can ask the one
# question that matters — does a stray separator change the canonical value —
# without re-pinning eight canonical forms other tests already own. Both
# policy rules are represented: `name` and `address` are SEPARATOR_ONLY, the
# other six EVERY_PART.
ONE_VALUE = {
    "name": "Коваленко Іван Петрович",
    "address": "вул. Шевченка, 1, Київ",
    "identifier": "1025400524313",
    "country": "ua",
    "date": "17.09.1980",
    "phone": "0501234567",
    "email": "Ivan@Example.com",
    "url": "https://a.example/page",
}


def test_a_stray_separator_does_not_survive_into_the_canonical_value():
    """Leading, trailing, doubled and whitespace-padded, on every type in the
    policy table.

    The split was never the bug: «Іван;» has ONE non-empty part, so
    `_canon_multivalue` declined it and the cell fell through to its ordinary
    canonicalizer, which kept the separator. `_binding_source` then told the
    engine to split that column on `;` and the engine read two values out of a
    cell holding one — measured end to end as `values: 1, emitted: 2,
    unaccounted: -1`. A validated type reached the same state by a longer
    route: `identifier` canonicalized «17.09.1980;» to «17091980;», a `;`
    inside a value the engine had been told to split.

    Asserted against the clean value's own canonical form rather than a
    literal, so the test states the property — debris must not change the
    answer — instead of a table of expected strings.
    """
    from ftmap.normalize.canonical import MULTIVALUE_POLICY

    assert set(ONE_VALUE) == set(MULTIVALUE_POLICY)
    for type_name, clean in ONE_VALUE.items():
        want = canonicalize(clean, type_name)
        assert want.ok is True, type_name
        for text in (f"{clean};", f";{clean}", f"{clean}; ", f" ;{clean}",
                     f"{clean};;", f";;{clean};;", f" ; {clean} ; "):
            got = canonicalize(text, type_name)
            assert got.ok is True, (type_name, text)
            assert got.value == want.value, (type_name, text)


def test_no_canonical_value_ever_splits_into_an_empty_part():
    """The invariant `_binding_source` relies on, asserted where it is made.

    `build.emit.coverage` counts a cell's values as the non-empty parts of its
    canonical form while the mapping engine splits the same string natively,
    so a canonical value with an empty part makes those two disagree and
    `unaccounted` goes negative. Checked here rather than only at the
    consumers: filtering empties on both sides would hide one bug in two
    places instead of fixing it.
    """
    from ftmap.normalize.canonical import MULTIVALUE_JOIN, MULTIVALUE_POLICY

    for type_name, clean in ONE_VALUE.items():
        assert type_name in MULTIVALUE_POLICY
        for text in (clean, f"{clean};", f";{clean}", f"{clean};;",
                     f" ; {clean} ; ", f"{clean};{clean}",
                     f";{clean};;{clean};"):
            got = canonicalize(text, type_name)
            if not got.ok:
                continue
            parts = (got.value or "").split(MULTIVALUE_JOIN)
            assert all(p.strip() for p in parts), (type_name, text, parts)


def test_debris_around_a_real_pack_costs_none_of_its_values():
    """Stripping the empty parts must not cost a part that held something."""
    assert canonicalize(";Іван;;Петро;", "name").value == "Іван;Петро"
    assert canonicalize(";Іван;;Петро;", "name").canonicalizer == "namex2"
    assert canonicalize(" ; ua ; pl ; ", "country").value == "ua;pl"
    assert canonicalize("1952-08-20;;2002-05-07;", "date").value == \
        "1952-08-20;2002-05-07"


def test_a_cell_holding_nothing_but_separators_is_refused_not_emptied():
    """DECIDED, AND THIS TEST IS THE RECORD OF IT: «;» and «;;» canonicalize to
    a refusal with a reason, not to «;» and not to an empty string.

    Both alternatives drop something in silence, which is the one thing this
    pipeline does not do. Accepted as «;», the cell is a filled cell worth
    zero values — `coverage` counts non-empty parts, finds none — so the
    source's own content leaves the accounting with nothing said about it.
    Canonicalized to «», it would be indistinguishable from a blank cell,
    which is a different fact about the source. Refused, it is one value and
    one reject line, the arithmetic closes at `unaccounted == 0`, and how many
    such cells a column held is a number a reader can see.

    The reason is value-free — `RejectTally` publishes these strings in
    `summary.json`, the one artefact allowed to leave the machine — so it
    names the policy and never the cell.
    """
    from ftmap.normalize.canonical import MULTIVALUE_POLICY

    for type_name in MULTIVALUE_POLICY:
        for text in (";", ";;", " ; ", " ;; "):
            got = canonicalize(text, type_name)
            assert got.ok is False, (type_name, text)
            assert got.value is None
            assert got.canonicalizer == f"{type_name}x0"
            assert "separator" in got.reason
            assert text.strip() not in got.reason


# A part that CANNOT be a value of its type, one per row of the policy table.
# EVERY_PART types get something their validator refuses; the two
# SEPARATOR_ONLY types have no validator to refuse anything, so their junk is
# a part with no letter or digit in it, which is the only thing that rule can
# still check. Both are asserted below rather than assumed.
UNCANONICAL_PART = {
    "name": "-",
    "address": "-",
    "identifier": "ab",
    "country": "zzzz",
    "date": "abc",
    "phone": "abc",
    "email": "abc",
    "url": "abc",
}


def test_a_bad_value_inside_a_real_pack_costs_only_itself():
    """«12;ab» is not two identifiers — `ab` has no digit — but the `;` is the
    publisher saying the cell holds several values, so `ab` is a bad value
    inside a real pack and `12` is still a value.

    The cell used to be accepted WHOLE: `_canon_identifier` strips `-–—./` and
    not `;`, so it came back as «12;ab», `_binding_source` told the engine to
    split that column on `;`, and the engine emitted both parts — including
    the one the guard had refused. Nothing reported it, which is why this is a
    test and not a number: `build.emit.coverage` splits the canonical value
    the same way the engine does, so both sides counted two and `unaccounted`
    stayed 0.

    Refusing the whole cell instead would have cost the values beside the bad
    part — 384 real identifiers on `ua_war_sanctions.targets.simple.csv` to
    suppress 192 letters-only tokens. The bad part is refused ON ITS OWN, with
    its own reason, and `refused` is how it reaches the reject line.
    """
    got = canonicalize("12;ab", "identifier")
    assert got.ok is True
    assert got.value == "12"
    assert got.canonicalizer == "identifierx1of2"
    assert [part for part, _ in got.refused] == ["ab"]
    # The part's own verdict, not a new opinion invented for the pack.
    assert got.refused[0][1] == canonicalize("ab", "identifier").reason
    # The other two canonicalizers that pass their text through. FtM's own url
    # validator accepts the packed form whole, and `_canon_free` is whitespace
    # normalization, so neither would have removed the separator either.
    assert canonicalize("https://a.example/x;abc", "url").value == \
        "https://a.example/x"
    assert canonicalize("Коваленко Іван; - ", "name").value == "Коваленко Іван"
    # And a pack every part of which canonicalizes is untouched, as is a
    # single value that merely contains an internal `/`.
    assert canonicalize("12;34", "identifier").canonicalizer == "identifierx2"
    assert canonicalize("12345/67", "identifier").value == "1234567"


def test_a_pack_no_part_of_which_is_a_value_is_refused_whole():
    """Partial acceptance has nothing to keep here, so the cell falls through
    and gets ONE reason about itself rather than n reasons about its pieces —
    and the fall-through may still not hand back the separator.

    «-;+» is the case that reaches both halves: `_canon_free` accepts it, so
    without the guard in `canonicalize` the engine would have been told to
    split a value holding no value at all.
    """
    got = canonicalize("-;+", "name")
    assert got.ok is False
    assert got.refused == ()
    assert "separator" in got.reason
    assert canonicalize("ab;cd", "identifier").ok is False
    assert canonicalize("ab;cd", "identifier").refused == ()


def test_the_reason_a_refused_part_gives_is_value_free():
    """`RejectTally` publishes these strings in `summary.json`, the one
    artefact allowed to leave the machine, so the reason may name the policy
    and never the cell. The part itself goes in the reject line's `value`
    field, which is owner-only, exactly like the whole cell's does."""
    got = canonicalize("1025400524313;Коваленко", "identifier")
    assert got.ok is True
    (part, reason), = got.refused
    assert part == "Коваленко"
    for value_word in ("1025400524313", "Коваленко"):
        assert value_word not in reason
    # The SEPARATOR_ONLY rule has no validator to borrow a reason from, so its
    # reason is written by this module and must hold to the same rule.
    (_, debris_reason), = canonicalize("Іван; - ; Петро", "name").refused
    assert "Іван" not in debris_reason and "-" not in debris_reason


def test_a_canonical_value_never_splits_into_a_part_that_is_not_a_value():
    """THE INVARIANT `_binding_source` RELIES ON, asserted where it is made.

    The engine is told to split a multi-valued column on `MULTIVALUE_JOIN`, so
    every part of an accepted canonical value is a value the deliverable will
    carry. What makes that true is the guard the split was admitted by, and
    the guard is only as good as the fall-through beneath it: a refused split
    that came back through the ordinary canonicalizer with the separator
    intact broke this while `coverage` — which splits the same string the same
    way — went on reporting `unaccounted: 0`.

    Stated as a property over the whole policy table rather than as a list of
    expected strings, so a row added to the table is covered by construction.
    """
    from ftmap.normalize.canonical import (EVERY_PART, MULTIVALUE_JOIN,
                                           MULTIVALUE_POLICY)

    assert set(UNCANONICAL_PART) == set(MULTIVALUE_POLICY)
    for type_name, junk in UNCANONICAL_PART.items():
        _, guard = MULTIVALUE_POLICY[type_name]
        # The junk really is junk, by whichever standard the row's rule uses.
        if guard == EVERY_PART:
            assert canonicalize(junk, type_name).ok is False, type_name
        else:
            assert not any(c.isalnum() for c in junk), type_name
        clean = ONE_VALUE[type_name]
        for text in (clean, f"{clean}{MULTIVALUE_JOIN}{clean}",
                     f"{clean};{junk}", f"{junk};{clean}",
                     f"{clean};{junk};{clean}", f";{clean};{junk};",
                     f"{clean};{clean};{junk}"):
            got = canonicalize(text, type_name)
            if not got.ok:
                continue
            for part in (got.value or "").split(MULTIVALUE_JOIN):
                assert part.strip(), (type_name, text, got.value)
                if guard == EVERY_PART:
                    assert canonicalize(part, type_name).ok, \
                        (type_name, text, part)
                else:
                    assert any(c.isalnum() for c in part), \
                        (type_name, text, part)
            # AND EVERY PART OF THE CELL IS SOMEWHERE. Partial acceptance
            # keeps the invariant above by removing the bad parts from the
            # value, so the other half of it is that the removed ones are
            # reported rather than dropped: what the source held is the parts
            # of the value plus the parts of `refused`, which is the identity
            # `coverage` publishes as `values == emitted + rejected`.
            kept = len([p for p in (got.value or "").split(MULTIVALUE_JOIN)
                        if p])
            assert kept + len(got.refused) == len(
                [p for p in MULTIVALUE_POLICY[type_name][0].split(text)
                 if p and p.strip()]), (type_name, text, got)


def test_a_separator_we_only_guessed_at_stays_all_or_nothing():
    """THE COUNTER-CASE TO PARTIAL ACCEPTANCE, and the reason `PURE_PACKERS`
    is a measurement rather than "any delimiter".

    Where the separator is one the publisher wrote to mean packing, a failing
    part is a bad value. Where it is one WE read as packing, every part
    canonicalizing was the only evidence for the split at all, so a failing
    part is evidence against the split and keeping the rest silently
    rewrites the cell:

      «АА 123456»  is one passport number written with a space. Partial
                   acceptance would publish `123456` and reject «АА».
      «Korea, Democratic People's Republic of»  is one country, inverted
                   round a comma the way FtM's own long forms are. Partial
                   acceptance publishes `kr` — the WRONG Korea — and rejects
                   the rest of its own name. Measured over the inverted forms:
                   `kp`->`kr`, `cd`->`cg`, `vg`->`gb`, each a silently wrong
                   value where the whole-cell fall-through is right.

    A wrong value with no reject line beside it is worse than the packed cell
    partial acceptance exists to unpack, so the guessed separators keep the
    all-or-nothing guard they always had.
    """
    # «АА 123456» is one passport number written with a space, and since
    # 2026-08-29 that is exactly how it canonicalizes — one identifier, not a
    # split and not a refusal. The guessed separator still does not license a
    # PARTIAL acceptance, which is what this test is about: the cell is read
    # whole or not at all.
    whole = canonicalize("АА 123456", "identifier")
    assert whole.ok and whole.value == "АА123456" and whole.refused == ()
    for text, code in (("Korea, Democratic People's Republic of", "kp"),
                       ("Congo, The Democratic Republic of the", "cd"),
                       ("Virgin Islands, British", "vg"),
                       ("Korea, Republic of", "kr")):
        got = canonicalize(text, "country")
        assert got.value == code, text
        assert got.refused == ()
    # A cell that mixes the two is judged by the weaker of its separators.
    assert canonicalize("12345;ab cd", "identifier").ok is False
    # None of which stops a guessed separator from splitting a cell whose
    # every part IS a value — that is what the all-or-nothing guard admits.
    assert canonicalize("380486852018 380486864319", "phone").canonicalizer \
        == "phonex2"
    assert canonicalize("380486852018, 380486864319", "phone").canonicalizer \
        == "phonex2"


def test_only_a_measured_packer_licenses_partial_acceptance():
    """Stated over the whole policy table rather than for `identifier` alone:
    a separator outside `PURE_PACKERS` never produces a partly accepted cell,
    whichever row splits on it, and both packers do wherever they split."""
    from ftmap.normalize.canonical import (MULTIVALUE_PACKS, MULTIVALUE_POLICY,
                                           PURE_PACKERS)

    assert PURE_PACKERS == ";|"
    for type_name, junk in UNCANONICAL_PART.items():
        splitter, _ = MULTIVALUE_POLICY[type_name]
        clean = ONE_VALUE[type_name]
        # A type with a pack of its own (`MULTIVALUE_PACKS`) has said which
        # characters its publisher packs with; those license partial
        # acceptance for that type as `PURE_PACKERS` do for every type.
        own = MULTIVALUE_PACKS.get(type_name)
        for sep in (",", "/", " ", ";", "|"):
            text = f"{clean}{sep}{junk}"
            if len([p for p in splitter.split(text) if p and p.strip()]) < 2:
                continue  # this row does not split on this character
            got = canonicalize(text, type_name)
            measured = sep in PURE_PACKERS or (
                own is not None and len(own[0].split(text)) > 1 and sep.strip())
            if measured:
                assert got.refused, (type_name, sep, got)
            else:
                assert got.refused == (), (type_name, sep, got)


# --------------------------------------------------------------------------
# Two date forms the pack did not have, both found by the same query: of 55
# columns where the etalon named a property and nothing was bound, 18 were the
# validator refusing the model's answer, and seven of those were a separator.
# --------------------------------------------------------------------------

def test_a_space_separated_timestamp_is_a_date():
    """RFC 3339 says `T`; SQL exporters overwhelmingly write a space. Measured
    on the ЄДРСР court-decisions register: four columns of
    `2025-12-31 00:00:00+02` canonicalized at 0.00 against shape
    `dddd-dd-dd dd:dd:dd+dd`, and the adjudication, receipt and publication
    dates were all refused for a separator."""
    assert canonicalize("2025-12-31 00:00:00+02", "date").value == "2025-12-30T22:00:00"
    assert canonicalize("2026-08-19T05:41:01", "date").value == "2026-08-19T05:41:01"


def test_an_offset_may_be_hours_only():
    """Postgres writes a whole-hour offset as `+02`, not `+02:00`. While the
    pattern demanded four digits this could not be reached; widening one
    without the other raised `int('')` and turned four correct bindings into a
    date-validator error."""
    assert canonicalize("2025-12-31 00:00:00+02", "date").ok
    assert canonicalize("2025-12-31 00:00:00-05:30", "date").value == "2025-12-31T05:30:00"
    assert canonicalize("2025-12-31 00:00:00Z", "date").value == "2025-12-31T00:00:00"


def test_an_alphabetic_month_is_a_date():
    """`12-MAR-2015` is Oracle's default and what every ICIJ leak was exported
    through: `incorporation_date`, `start_date` and `end_date` all scored 0.00
    or below a quarter against shape `dd-LLL-dddd`."""
    assert canonicalize("12-MAR-2015", "date").value == "2015-03-12"
    assert canonicalize("1-jan-1999", "date").value == "1999-01-01"
    assert canonicalize("07-September-2020", "date").value == "2020-09-07"


def test_a_month_that_is_not_a_month_is_refused_not_guessed():
    """The alphabetic pattern matches any three-to-nine letters, so the month
    map is what decides. An unrecognised name must fall through to a refusal
    rather than become a plausible date."""
    for bad in ("12-XXX-2015", "12-FOO-2015", "12-M-2015"):
        assert not canonicalize(bad, "date").ok, bad


def test_the_new_patterns_do_not_swallow_a_plain_number():
    """`2006.0` and a bare engine displacement must still be refused: the
    reason the pack is conservative is that a number inside a plausible year
    range half-passes and the acceptance floor sees a middling score."""
    for bad in ("2006.0", "1956", "12345"):
        assert not canonicalize(bad, "date").ok, bad


def test_a_city_is_not_a_country_even_when_rigour_resolves_it():
    """`rigour` resolves «Київ» to `ua-30`, an ISO 3166-2 subdivision, and this
    canonicalizer accepted it for a `country` property followthemoney will not
    store — so the value produced no statement, no reject line and no
    explanation. That is the whole of the 38-value residual carried as "not yet
    explained" since the first measurement: both ship registers publish
    `Україна, Київ` in their country column, the comma splits it, and the
    second half vanished."""
    assert canonicalize("Київ", "country").ok is False
    assert "subdivision" in canonicalize("Київ", "country").reason
    # The whole cell is refused, not half-kept. A comma is an ambiguous
    # separator in a country column — `Korea, Republic of` and `Bonaire, Sint
    # Eustatius and Saba` are country names containing one — so partial
    # acceptance stays restricted to the unambiguous packers, and an ambiguous
    # cell is refused with a reason rather than guessed at.
    whole = canonicalize("Україна, Київ", "country")
    assert not whole.ok
    # The reason it carries is the generic one — the aggregate path reports
    # that the cell as a whole did not resolve, not which half failed and why.
    # Coarser than the single-value case above, and still value-free and on
    # record, which is what the accounting needs.
    assert whole.reason and "Київ" not in whole.reason


def test_a_real_country_still_resolves():
    for v, code in (("Україна", "ua"), ("США", "us"), ("Польща", "pl")):
        r = canonicalize(v, "country")
        assert r.ok and r.value == code, v


def test_a_latin_code_without_a_digit_is_an_identifier():
    """The digit rule exists to stop a surname column scoring 1.00 as
    `identifier`, and it also refuses codes that are genuinely identifiers.

    Measured on the person-graph corpus 2026-08-29: the model chose the
    etalon's own answer and this gate threw it away — `Airplane:
    registrationNumber` on the aircraft register (`UR-ALUR`, acceptance 0.10)
    and `LegalEntity:programId` on the war-sanctions file (`UA-WS-MILIND`,
    0.00). Both are Latin codes carrying a separator, which is what a surname
    never is.
    """
    for value in ("UR-ALUR", "UA-WS-MILIND", "NK-ABKNEMMRUsrpnXYqoiYiyb"):
        assert canonicalize(value, "identifier").ok, value


def test_a_word_without_a_digit_is_still_not_an_identifier():
    """The four columns the digit rule was written for: a surname, a party
    name, a street name and a free-text biography."""
    for value in ("ЩЕРБАК", "Коваленко", "Слуга народу", "Хрещатик"):
        assert not canonicalize(value, "identifier").ok, value


def test_a_series_and_number_written_with_a_space_is_one_identifier():
    """`АК 00075` — a Ukrainian certificate series and its number. The
    canonicalizer's own docstring said no such column had been seen in either
    corpus; the land valuers register is 2 881 of them, and it was only
    invisible because the file was being read as cp1251."""
    result = canonicalize("АК 00075", "identifier")
    assert result.ok and result.value == "АК00075"


def test_a_sentence_is_not_an_identifier_however_many_digits_it_has():
    assert not canonicalize("Постанова від 12 січня 2024 року", "identifier").ok


def test_a_sentinel_does_not_count_against_a_column_s_acceptance():
    """`XXX` is a null placeholder, not a country, and the profile already
    measures the share of a column that is one. Counted as a failing value it
    took `Company:jurisdiction` on the ICIJ entities file to 0.86 against a
    0.90 floor — a correct binding refused by four points of placeholder."""
    rate, _ = acceptance(["Ukraine", "Cyprus", "XXX", "Undetermined"], "country")
    assert rate == 1.0


def test_an_all_zero_date_is_a_null_placeholder():
    """`0000-00-00` is what a MySQL export writes for a date that was never
    set; it is not a date and no canonicalizer will ever make one of it.
    Measured on the ship register's deregistration column: 25 % of the
    filled cells, acceptance 0.74 against a 0.90 floor, and the model shown
    `0000-00-00` as its first sample declined the column outright."""
    assert is_sentinel("0000-00-00")
    assert is_sentinel("0000-00-00 00:00:00")
    assert is_sentinel("00.00.0000")
    assert is_sentinel("0000")
    assert not is_sentinel("2000-01-01")
    assert not is_sentinel("0")
    rate, rejected = acceptance(["0000-00-00", "2019-01-02", "07.12.2007"], "date")
    assert rate == 1.0 and rejected == []


def test_an_all_sentinel_column_cannot_buy_acceptance():
    """Filtering sentinels from the denominator must not make a column of
    nothing but placeholders score 1.0 against any floor: such a binding
    asserts nothing and every value of it dies at emission."""
    share, rejected = acceptance(["-", "--", "XXX", "не визначено"], "name")
    assert share == 0.0
    assert rejected
    # A genuinely empty column is still not a contradiction.
    assert acceptance([], "identifier") == (1.0, [])
    assert acceptance([None, ""], "identifier") == (1.0, [])


def test_a_two_token_identifier_is_a_series_and_a_number_not_a_phrase():
    """«АК 00075» is one certificate; «кв. 12» is an address fragment. Any
    2-token digit-bearing string passing let address and case-number prose
    canonicalize as identifiers."""
    assert canonicalize("АК 00075", "identifier").value == "АК00075"
    for phrase in ["кв. 12", "справа 456/2024", "Suite 100"]:
        assert canonicalize(phrase, "identifier").ok is False, phrase


def test_a_digitless_code_is_a_code_in_any_alphabet_and_a_name_in_none():
    """UR-ALUR and UA-WS-MILIND are codes; Mary-Jane is a name. The old rule
    said 'a code is ASCII', which passed the name and would refuse the same
    shape written in Cyrillic."""
    for code in ["UR-ALUR", "UA-WS-MILIND", "АБ-ВГ"]:
        assert canonicalize(code, "identifier").ok is True, code
    for name in ["Mary-Jane", "Аль-Каїда"]:
        assert canonicalize(name, "identifier").ok is False, name


def test_a_placeholder_written_on_two_lines_is_still_a_placeholder():
    """The aircraft register's operator column: «Не відноситься» over
    "Not applicable" in one cell, 588 of 872 rows. Every line a placeholder
    is a placeholder; a cell that says something on any line is not."""
    assert is_sentinel("Не відноситься\nNot applicable")
    assert is_sentinel("  -  \n  n/a ")
    assert not is_sentinel("Не відноситься\nТОВ Ромашка")
    assert not is_sentinel("Not applicable to vessels")


def test_no_party_is_a_placeholder_not_a_party():
    from ftmap.normalize.canonical import is_sentinel
    assert is_sentinel("Безпартійний") and is_sentinel("безпартійна")
    assert not is_sentinel("Політична партія «Слуга Народу»")


def test_a_run_of_asterisks_is_a_redaction_and_a_null():
    from ftmap.normalize.canonical import is_sentinel
    assert is_sentinel("**********") and is_sentinel("***")
    assert not is_sentinel("**") and not is_sentinel("*1234*")


def test_an_office_phone_list_is_split_on_its_commas_not_its_spaces():
    """The enforcement offices write «(04142) 3-08-30, 3-00-07» — one full
    number and a local extension under the same exchange — and the lease
    register «(044) 123-45-67, 123-45-68». Split on whitespace, «(04142)»
    is a part and the cell fails (`work-c12`). A comma, a semicolon, a slash
    and a line break are packers no phone number contains, so they are
    tried first and whole; a local part that is shorter than a national
    number takes the exchange of the number before it."""
    got = canonicalize("(04142) 3-08-30, 3-00-07", "phone")
    assert got.ok and got.value == "+380414230830;+380414230007"
    assert got.canonicalizer == "phonex2"
    got = canonicalize("(044) 123-45-67; 123-45-68\n(044) 123-45-69", "phone")
    assert got.value == "+380441234567;+380441234568;+380441234569"
    # A local number with nothing before it to complete it is not a phone.
    assert canonicalize("123-45-67, 123-45-68", "phone").ok is False
    # A part that is not a number at all is refused beside the real one.
    got = canonicalize("(04142) 3-08-30, дзвонити після 18", "phone")
    assert got.ok and got.value == "+380414230830" and len(got.refused) == 1
    # Whitespace-only packs still work as before.
    assert canonicalize("380486852018 380486864319", "phone").canonicalizer == "phonex2"


def test_a_label_before_the_number_is_the_publisher_s_and_is_stripped():
    """A mobile-forensics export writes every entry of a contact's number
    block as `<label>: <number>`, several to a cell on their own lines. The
    label is the publisher's; a trailing note is still not."""
    got = canonicalize("Mobile-: +380501234567", "phone")
    assert got.ok is True and got.value == "+380501234567"
    got = canonicalize("Mobile-: +380501234567\nHome-Fax: 0442345678", "phone")
    assert got.ok is True
    assert got.value == "+380501234567;+380442345678"
    assert got.canonicalizer == "phonex2"
    assert canonicalize("0501234567 не телефон", "phone").ok is False
    assert canonicalize("Коваленко Іван: 0501234567", "phone").ok is True
