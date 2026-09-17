# tests/test_labels.py
"""The Ukrainian label catalogue, and how a per-schema qname finds it."""

from ftmap.vocab.catalogue import Catalogue
from ftmap.vocab.labels import Labels

CAT = Catalogue.load()
LAB = Labels.load()


def test_an_exact_qname_is_used_as_written():
    assert LAB.label_uk("Person:birthDate")


def test_an_inherited_property_resolves_to_its_declaring_schema():
    """`Person:name` is `Thing:name` — one Property declared on Thing."""
    assert LAB.label_uk("Thing:name") is not None, "two Nones would pass vacuously"
    assert LAB.label_uk("Person:name") == LAB.label_uk("Thing:name")


def test_an_ambiguous_name_resolves_by_ancestor_when_it_can():
    """`title` and `registrationNumber` carry two labels each; the schema
    asked about is what tells them apart."""
    assert LAB.label_uk("Person:title") != LAB.label_uk("Document:title")
    assert LAB.label_uk("Vehicle:registrationNumber") != LAB.label_uk(
        "Company:registrationNumber")


def test_metadata_keys_are_not_properties():
    assert LAB.label_uk("_about") is None
    assert LAB.label_uk("_vendored_from") is None


def test_an_unknown_property_has_no_label_rather_than_a_wrong_one():
    assert LAB.label_uk("Person:notAProperty") is None


def test_an_ambiguous_name_with_no_ancestor_match_is_left_unlabelled():
    """`title` carries two different labels in the file. Guessing between
    them would put a wrong Ukrainian gloss on a real property, which is worse
    than falling back to the English one."""
    assert LAB.label_uk("Vehicle:notAProperty") is None


def test_the_hint_is_available_when_the_file_has_one():
    assert LAB.hint("Thing:name")
