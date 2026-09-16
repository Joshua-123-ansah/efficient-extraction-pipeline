"""
Author-name handling.

These helpers are duplicated across all four extraction pipelines, so every
test runs against all of them. A name that fails to match here means papers are
silently skipped and affiliations are dropped.
"""

import pytest

MODULES = ["joshcoley", "grobid", "llm", "npl", "task1"]


@pytest.fixture(params=MODULES)
def mod(request, extractors):
    return extractors[request.param]


# ---------------------------------------------------------------- suffixes

@pytest.mark.parametrize("raw,expected", [
    ("James Holly, Jr.", "James Holly"),
    ("James Holly Jr", "James Holly"),
    ("Ann Bee, PhD", "Ann Bee"),
    ("Ann Bee III", "Ann Bee"),
    ("Ann Bee, Jr., PhD", "Ann Bee"),
    ("Ann Bee", "Ann Bee"),
])
def test_strip_name_suffix(mod, raw, expected):
    assert mod.strip_name_suffix(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("James Holly, Jr.", ("James", "Holly")),
    ("Cox, Monica F.", ("Monica", "Cox")),
    ("Brooke Coley", ("Brooke", "Coley")),
    ("Madonna", ("Madonna", "")),
    ("", ("", "")),
])
def test_extract_first_last(mod, raw, expected):
    assert mod.extract_first_last(raw) == expected


# ------------------------------------------------------------ normalisation

@pytest.mark.parametrize("raw,expected", [
    ("Na'ilah", "nailah"),
    ("Boyd-Sinkler", "boydsinkler"),
    ("Résumé", "resume"),
    ("O'Brien", "obrien"),
    ("", ""),
])
def test_normalize_name_for_compare_robust(mod, raw, expected):
    assert mod.normalize_name_for_compare_robust(raw) == expected


@pytest.mark.parametrize("raw,norm,expected", [
    ("B.", "b", True),
    ("B", "b", True),
    ("B.A.", "ba", True),
    ("B. A.", "ba", True),
    ("Bo", "bo", False),
    ("Brooke", "brooke", False),
])
def test_is_initials(mod, raw, norm, expected):
    assert mod._is_initials(raw, norm) is expected


# ---------------------------------------------------------------- matching

@pytest.mark.parametrize("target,found", [
    ("Brooke Coley", "Brooke Coley"),
    ("Brooke Coley", "B. Coley"),
    ("Brooke Coley", "B.A. Coley"),
    ("Brooke Coley", "B. A. Coley"),
    ("Denise R. Simmons", "D. R. Simmons"),
    ("Monica Cox", "Monica F Cox"),
    ("Ebony O. McGee", "Ebony McGee"),
    ("James Holly, Jr.", "James Holly"),
    ("Na'ilah Suad Nasir", "Nailah Nasir"),
    ("Karis Boyd-Sinkler", "K. Boyd Sinkler"),
    ("Walter Lee", "Walter C. Lee"),
])
def test_author_match_positive(mod, target, found):
    assert mod.robust_author_match(target, found) is True
    assert mod.robust_author_match(found, target) is True, "matching must be symmetric"


@pytest.mark.parametrize("target,found", [
    ("Brooke Coley", "Michael Coley"),
    ("Brooke Coley", "Brooke Smith"),
    ("Kelly Cross", "David Cross"),
    ("Bo Lee", "Brooke Lee"),
    ("Ann Bee", ""),
    ("", "Ann Bee"),
    ("", ""),
    ("Madonna", "Madonna"),          # no surname -> cannot confirm identity
])
def test_author_match_negative(mod, target, found):
    assert mod.robust_author_match(target, found) is False


# ------------------------------------------------------------- abbreviation

@pytest.mark.parametrize("raw,expected", [
    ("Monica Lynn Miles", "Monica L. Miles"),
    ("Walter Lee", "Walter Lee"),
    ("Cox, Monica F.", "Monica F. Cox"),
    ("James Holly, Jr.", "James Holly"),
    ("", ""),
])
def test_abbreviate_target_name(mod, raw, expected):
    assert mod.abbreviate_target_name(raw) == expected


def test_abbreviated_target_still_matches_byline(mod):
    """
    Abbreviation runs before matching. If it mangles the name, the paper is
    lost even though robust_author_match would have handled the raw form.
    """
    for raw, byline in [
        ("James Holly, Jr.", "James Holly"),
        ("Monica Lynn Miles", "Monica L. Miles"),
        ("Denise R. Simmons", "D. R. Simmons"),
    ]:
        target = mod.abbreviate_targets([raw])[0]
        assert mod.robust_author_match(target, byline) is True, f"{raw!r} -> {target!r}"


def test_abbreviate_targets_preserves_count(mod):
    names = ["Brooke Coley", "James Holly, Jr.", "Monica Lynn Miles"]
    assert len(mod.abbreviate_targets(names)) == len(names)
