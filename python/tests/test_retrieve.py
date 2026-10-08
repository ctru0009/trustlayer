"""Phase 5: ACL predicate, seeded assignment, filtered search, leak audit."""

import os

import pytest

from trustlayer.retrieve.acl import USERS, visible
from trustlayer.retrieve.seed import assign

psycopg = pytest.importorskip("psycopg", reason="needs the embed extra")

DB_URL = os.environ.get("DATABASE_URL")
needs_db = pytest.mark.skipif(not DB_URL, reason="needs DATABASE_URL")


def test_confidential_needs_explicit_grant() -> None:
    assert visible("confidential", ["HR"], ["HR", "Everyone"])
    assert not visible("confidential", ["Everyone"], ["Everyone"])
    assert not visible("confidential", ["HR"], ["Finance", "Everyone"])


def test_public_and_internal_need_any_shared_role() -> None:
    assert visible("public", ["Everyone"], ["Everyone"])
    assert visible("internal", ["Everyone", "HR"], ["Finance", "Everyone"])
    assert not visible("internal", ["HR"], ["Finance", "Everyone"])
    assert not visible("public", [], ["Everyone"])


def test_everyone_only_user_never_sees_confidential() -> None:
    carol = list(USERS["carol"])
    assert not visible("confidential", ["Everyone"], carol)
    assert not visible("confidential", ["HR", "Everyone"], carol)
    assert not visible("confidential", ["Finance"], carol)


def test_seed_assignment_is_deterministic() -> None:
    first = assign("train:abc", "internal", "text")
    assert assign("train:abc", "internal", "text") == first
    assert set(first) == {"Everyone", first[1]} and first[1] in ("HR", "Finance")


def test_seed_rules_per_label() -> None:
    assert assign("d", "public", "text") == ["Everyone"]
    assert assign("funsd:1", "internal", "image") == ["Everyone"]
    conf = assign("d", "confidential", "text")
    assert len(conf) == 1 and conf[0] in ("HR", "Finance")


@needs_db
def test_search_returns_only_visible_hits() -> None:
    from trustlayer.retrieve.search import search

    conn = psycopg.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM chunks LIMIT 1")
            row = cur.fetchone()
        assert row is not None
        vec = list(row[0])
        for user in USERS:
            hits = search(conn, vec, user, top_k=10)
            assert len(hits) <= 10
            for hit in hits:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT label, allowed_roles FROM documents WHERE id = %s",
                        (hit.doc_id,),
                    )
                    label, allowed = cur.fetchone()
                ok = visible(label, list(allowed), list(USERS[user]))
                assert ok, (user, hit.doc_id)
    finally:
        conn.close()


@needs_db
def test_search_unknown_user_raises() -> None:
    from trustlayer.retrieve.search import search

    conn = psycopg.connect(DB_URL)
    try:
        with pytest.raises(KeyError):
            search(conn, [0.0] * 768, "mallory", top_k=1)
    finally:
        conn.close()
