"""Unit tests for trustlayer.prep.clean (pure Python, no Spark)."""

from trustlayer.prep.clean import (
    clean_text,
    normalize_subject,
    thread_id_from_subject,
)


def test_clean_strips_quoted_lines() -> None:
    body = "Here is the update.\n> On Monday you wrote:\n> please send it\nThanks!"
    assert clean_text(body) == "Here is the update.\nThanks!"


def test_clean_cuts_signature() -> None:
    body = "Hello team,\nsee attached.\n-- \nJane Doe\nVP of Things"
    assert clean_text(body) == "Hello team,\nsee attached."


def test_clean_drops_forwarded_separator() -> None:
    body = "FYI below.\n-----Original Message-----\nhello again"
    assert clean_text(body) == "FYI below.\nhello again"


def test_clean_drops_attribution_line() -> None:
    body = "Agreed.\nOn Monday, Jane wrote:\nlet us meet"
    assert clean_text(body) == "Agreed.\nlet us meet"


def test_clean_collapses_blanks_and_strips() -> None:
    assert clean_text("  one  \n\n\n\ntwo \n") == "one\n\ntwo"


def test_clean_empty_stays_empty() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n  \n") == ""


def test_normalize_subject_strips_prefix_chains() -> None:
    assert normalize_subject("Re: Fw: RE:  Budget Q3 ") == "budget q3"


def test_thread_id_groups_replies() -> None:
    a = thread_id_from_subject("Re: Budget Q3")
    assert a == thread_id_from_subject("budget q3")


def test_thread_id_splits_topics() -> None:
    a = thread_id_from_subject("Budget Q3")
    assert a != thread_id_from_subject("Holiday party")
