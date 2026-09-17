from ftmap.normalize.canonical import canonicalize, column_evidence
from ftmap.profile.detectors import (DETECTORS, detect, is_email, is_inn_ru,
                                     is_numeric, is_url, is_proper_name,
                                     is_legal_name)


def test_ukrainian_written_dates():
    d = DETECTORS["date"]
    assert d("17.09.1980") is True
    assert d("1980-09-17") is True
    assert d("17/09/1980") is True
    assert d("не вказано") is False


def test_excel_serial_dates_detect_as_date_not_only_as_numeric():
    """I4: `is_date` had its own pattern list with no serial-date branch, so
    a column of Excel serial dates scored `{"numeric": 1.0}` and never
    retrieved a `date`-typed candidate, even though the canonicalizer accepted
    every value in it. `is_date` is now `canonicalize(v, "date", evidence).ok`
    over the same evidence, so the two can no longer disagree — including now
    that the evidence is what admits the serial at all."""
    serials = ["44927", "44928", "29481"]
    assert detect(serials)["date"] == 1.0
    ev = column_evidence(serials)
    for serial in serials:
        assert DETECTORS["date"](serial, ev) is True
        assert canonicalize(serial, "date", ev).ok is True


def test_a_measurement_column_does_not_detect_as_dates():
    """Measured on the МВС vehicle registry: CAPACITY (engine displacement,
    cc) detected `date` 0.27 and TOTAL_WEIGHT (kerb weight, kg) 0.28, so the
    shortlist offered date-typed properties for both and the model was shown a
    profile line saying a quarter of the column looked like dates. MAKE_YEAR,
    in the same file, IS a year column and must keep its 1.00."""
    displacement = ["1956", "2993", "1598", "1248", "1968", "2000", "998"]
    weight = ["1922", "2575", "1505", "1945", "990", "3500", "1240"]
    years = ["1949", "1956", "2015", "2013", "2026", "1980", "2001"]

    assert "date" not in detect(displacement)
    assert "date" not in detect(weight)
    assert detect(displacement)["numeric"] == 1.0
    assert detect(years)["date"] == 1.0
    assert detect(years)["numeric"] == 1.0


def test_the_date_detector_and_the_gate_read_one_column_the_same_way():
    """The divergence this module exists to prevent, in its newest form: the
    profile could offer date candidates for a column `acceptance()` then
    rejects wholesale. Both now derive the same `ColumnEvidence` from the same
    values."""
    from ftmap.normalize.canonical import acceptance

    for column in (["1956", "2993", "1598", "998"],
                   ["1949", "2015", "2026"],
                   ["44927", "44928"],
                   ["17.09.1980", "1975", "01.02.1990"]):
        assert detect(column).get("date", 0.0) == acceptance(column, "date")[0]


def test_phones_of_both_countries():
    assert DETECTORS["phone_ua"]("+380501234567") is True
    assert DETECTORS["phone_ua"]("(050) 123-45-67") is True
    assert DETECTORS["phone_ru"]("79253902086") is True
    assert DETECTORS["phone_ru"]("+380501234567") is False


def test_phone_detectors_never_accept_what_canonicalization_rejects():
    """I4: `is_phone_ua`/`is_phone_ru` now share `phone_rule_nsn` with
    `_canon_phone` instead of carrying their own regexes, so a value can no
    longer detect as a phone number that the canonicalizer then rejects. The
    concrete divergence measured before the fix: the old `is_phone_ru` regex
    (`d[0] in "78" and d[1] == "9"`) ignored a leading `+` entirely, so
    "+89253902086" — not a real international form, since Russia's country
    code is 7, not 8 — detected as True while `_canon_phone` rejected it."""
    assert DETECTORS["phone_ru"]("+89253902086") is False
    assert canonicalize("+89253902086", "phone").ok is False
    for value in ("89253902086", "+380501234567", "+77012345678",
                  "+89253902086", "не телефон", "123"):
        for detector_key in ("phone_ua", "phone_ru"):
            if DETECTORS[detector_key](value):
                assert canonicalize(value, "phone").ok, value


def test_ukrainian_identifiers_use_their_checksums():
    # Three real codes from the corpus, spanning both weight ranges.
    for code in ("22819278", "34549336", "05480654"):
        assert DETECTORS["edrpou"](code) is True, code
    assert DETECTORS["edrpou"]("2281927") is False
    # A 10-digit string with a wrong check digit is not an RNOKPP.
    assert DETECTORS["rnokpp"]("1234567890") is False


def test_row_ordinals_padded_to_eight_digits_are_not_edrpou():
    """The population the checksum exists to exclude."""
    for ordinal in ("00000001", "00000012", "00000123"):
        assert DETECTORS["edrpou"](ordinal) is False, ordinal


def test_country_accepts_names_and_explicit_codes_only():
    c = DETECTORS["country"]
    for name in ("Україна", "Росія", "Польща", "Республіка Білорусь", "UA", "GBR"):
        assert c(name) is True, name
    # Short tokens that collide with ISO codes: ordinary words, unit
    # abbreviations, N/A markers. A bare territory lookup accepts all of them.
    for token in ("no", "is", "to", "na", "id", "kg", "cm", "ua", "Іван"):
        assert c(token) is False, token


def test_russian_snils_checksum():
    assert DETECTORS["snils_ru"]("591903748103") is False  # 12 digits, not SNILS
    assert DETECTORS["snils_ru"]("11223344595") is True
    assert DETECTORS["snils_ru"]("11223344596") is False


def test_gender_codes_seen_in_this_corpus():
    g = DETECTORS["gender_code"]
    for v in ("ч", "ж", "м", "1", "2", "чол.", "жін."):
        assert g(v) is True
    assert g("Іван") is False


def test_bare_ascii_gender_letters_detect_too():
    """I4: `is_gender_code` had its own spelling set with no bare `m`/`f`,
    which GENDER_LEXICON (used for canonicalization) does carry, so a column
    of them detected nothing yet canonicalized cleanly."""
    g = DETECTORS["gender_code"]
    assert g("m") is True and canonicalize("m", "gender").ok is True
    assert g("f") is True and canonicalize("f", "gender").ok is True


def test_detect_returns_shares_and_drops_zeroes():
    got = detect(["17.09.1980", "01.02.1990", None, ""])
    assert got["date"] == 1.0
    assert "email" not in got


def test_is_email():
    assert is_email("ivan@example.com") is True
    assert is_email(" ivan@example.com ") is True  # surrounding whitespace stripped
    assert is_email("не email") is False
    assert is_email("ivan@example") is False  # no dot after the @


def test_is_url():
    assert is_url("https://example.com/page") is True
    assert is_url("http://example.com") is True
    assert is_url("example.com") is False  # no scheme
    assert is_url("не url") is False


def test_is_numeric():
    """Gates the `number` type into the shortlist (TYPE_FOR_DETECTOR) and had
    no test at all before this."""
    assert is_numeric("123") is True
    assert is_numeric("-123") is True
    assert is_numeric("123.45") is True
    assert is_numeric("123,45") is True
    assert is_numeric("не число") is False
    assert is_numeric("12.34.56") is False  # a date-shaped string, not a number


def test_is_inn_ru():
    # The stdnum library's own published examples: 12-digit personal and
    # 10-digit company ИНН, each with its checksum flipped to show the
    # rejection.
    assert is_inn_ru("123456789047") is True
    assert is_inn_ru("1234567894") is True
    assert is_inn_ru("123456789037") is False
    assert is_inn_ru("1234567895") is False
    assert is_inn_ru("не число") is False


def test_a_column_of_people_detects_as_a_name():
    """`name` is the most common FtM type in a person-graph corpus and the
    only one with no detector at all — date, phone, email, url, the four
    identifier checksums, gender, country and number each have one.

    Retrieval scores a property by lexical match on the header PLUS a type
    bonus from the detectors, so a name property could only ever be reached
    through its header. Measured on the person-graph corpus 2026-08-29: the
    court decisions' `judge` column was offered no name property among its 24
    candidates and the model bound `Person:gender`, which the engine then
    refused; the aircraft register's `Експлуатант/ Орендар` reached
    `Person:name` only at rank 19.
    """
    assert is_proper_name("ЩЕРБАК АНДРІЙ ІВАНОВИЧ")
    assert is_proper_name("Антипенко Ірина Вікторівна")
    assert is_proper_name("Коваленко Іван")
    assert is_proper_name("Petro Poroshenko")
    assert is_proper_name("Мельник-Ткаченко Ольга Петрівна")


def test_what_is_not_a_name_is_not_detected_as_one():
    """The detector earns its type bonus only if it refuses the columns that
    surround a name in these registers."""
    for value in ("12345678",                    # ЄДРПОУ
                  "31.12.2024",                  # a date
                  'ТОВ "РОМАШКА"',               # quotes: a legal form, not a person
                  "вул. Хрещатик, 1",            # an address
                  "Чинна",                       # a one-word status
                  "ivan@example.com",
                  "Товари",                      # a one-word category
                  "UA-WS-MILIND",                # an identifier
                  "Постанова Кабінету Міністрів України від 12 січня"):  # prose
        assert not is_proper_name(value), value


def test_a_column_of_organisations_detects_as_a_legal_name():
    """The commonest column of a Ukrainian register, and the one
    `is_proper_name` deliberately refuses: the quotes say legal form, not
    person. A legal-form abbreviation in front, or the name in any of the
    corpus's quotation marks, is a register writing an organisation."""
    for value in ('ТОВ "РОМАШКА"', "ПрАТ «Київстар»", 'ДП "УКРВОДШЛЯХ"',
                  "ФОП Коваленко Іван Петрович",
                  'ДЕРЖАВНЕ ПІДПРИЄМСТВО "АНТОНОВ"',
                  "RS AVIA LIMITED LIABILITY COMPANY", "Wind Rose LLC",
                  'КОМУНАЛЬНЕ ПІДПРИЄМСТВО „ВОДОКАНАЛ“'):
        assert is_legal_name(value), value


def test_what_is_not_an_organisation_is_not_a_legal_name():
    for value in ("Коваленко Іван Петрович", "Rotax", "12345678", "ТОВ",
                  "м. Київ, вул. Хрещатик, 1", "Дійсне\nValid", "",
                  'Наказ від 26.04.2023 № 315 "Про затвердження"'):
        assert not is_legal_name(value), value


def test_a_null_placeholder_is_not_a_value_a_detector_failed():
    """The aircraft register's operator column: 67 % «Не відноситься»,
    every real cell an organisation. The share is of the real values, as
    `acceptance()` already reads a column; a column of nothing but
    placeholders detects as nothing."""
    got = detect(["Не відноситься\nNot applicable", "Не відноситься\nNot applicable",
                  'ТОВ "РС АВІА"'])
    assert got["legal_name"] == 1.0
    assert detect(["-", "XXX", "не визначено"]) == {}


def test_an_organisation_s_name_is_read_by_the_words_it_is_made_of():
    from ftmap.profile.detectors import is_org_name
    assert is_org_name("Головне управління Національної поліції в Житомирській області")
    assert is_org_name('Політична партія "Слуга Народу"')
    assert is_org_name("Постійна депутатська комісія з питань бюджету, член комісії")
    assert is_org_name("Коростенський міськрайонний відділ державної виконавчої служби")
    assert is_org_name('ТОВ "Ромашка"')  # a legal form still counts
    assert not is_org_name("Коваленко Іван Петрович")
    assert not is_org_name("Виборчий округ № 11, Вінницька область")  # digits
    assert not is_org_name("Освіта вища. Постійна депутатська комісія з питань регламенту та депутатської діяльності, член комісії, з 2020 року")
    assert not is_org_name("майстер лісу")


def test_the_phone_detector_reads_one_number_per_cell_and_the_canonicalizer_reads_packs():
    """The detector's rate is what the prompt shows, and showing a pack-aware
    rate re-rolled two files worse twice (`work-c12`, `work-p1`); the pack
    is the canonicalizer's, read by `bind_by_detector` off the prompt. The
    engine keeps more than the detector promises, never less."""
    from ftmap.normalize.canonical import canonicalize
    assert DETECTORS["phone_ua"]("(04142) 3-08-30, 3-00-07") is False
    assert canonicalize("(04142) 3-08-30, 3-00-07", "phone").ok is True
    assert DETECTORS["phone_ua"]("123-45-67, 123-45-68") is False
    assert DETECTORS["phone_ua"]("Коваленко Іван") is False
