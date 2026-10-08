"""Permission-filtered vector search (spec section 8.4).

The permission predicate lives in the SQL itself (pre-filter), and
every returned row is re-checked in Python (``acl.visible``) before
returning — defence in depth. A mismatch between the two layers raises
instead of leaking.
"""

from __future__ import annotations

from dataclasses import dataclass

from trustlayer.retrieve.acl import USERS, visible


@dataclass(frozen=True)
class Hit:
    """One search result (chunk + owning document)."""

    chunk_id: str
    doc_id: str
    title: str
    snippet: str
    modality: str
    label: str
    distance: float


def search(conn: object, vector: list[float], user: str, top_k: int = 10) -> list[Hit]:
    """Top-k chunks by cosine distance, filtered to what ``user`` may see.

    Args:
        conn: psycopg connection (needs the ``embed`` extra).
        vector: 768-d normalized query vector.
        user: one of ``acl.USERS``.
        top_k: max results.

    Raises:
        KeyError: unknown user.
        PermissionError: SQL pre-filter and Python re-check disagree.

    """
    roles = list(USERS[user])
    vec = "[" + ",".join(repr(x) for x in vector) + "]"
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        # strict_order: filtered HNSW can return fewer than LIMIT rows
        # (spec §8.4.4 — observed: carol/q553 boilerplate query got 7/10).
        # Iterative scan re-walks until LIMIT passing rows are found.
        cur.execute("SET hnsw.iterative_scan = 'strict_order'")
        cur.execute(
            "SELECT c.id, c.doc_id, d.title, c.text, d.modality, d.label,"
            " c.embedding <=> %s::vector AS dist"
            " FROM chunks c JOIN documents d ON d.id = c.doc_id"
            " WHERE d.allowed_roles && %s::text[]"
            " AND (d.label != 'confidential' OR d.allowed_roles && %s::text[])"
            " ORDER BY c.embedding <=> %s::vector LIMIT %s",
            (vec, roles, [r for r in roles if r != "Everyone"], vec, top_k),
        )
        rows = cur.fetchall()
    hits = [
        Hit(chunk_id, doc_id, title, text[:200], modality, label, float(dist))
        for chunk_id, doc_id, title, text, modality, label, dist in rows
    ]
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(
            "SELECT id, label, allowed_roles FROM documents WHERE id = ANY(%s)",
            ([h.doc_id for h in hits],),
        )
        granted = {doc_id: (label, list(allowed)) for doc_id, label, allowed in cur}
    for hit in hits:
        label, allowed = granted[hit.doc_id]
        if not visible(label, allowed, roles):
            msg = f"ACL re-check failed: {hit.doc_id}"
            raise PermissionError(msg)
    return hits
