"""Seeded ACL assignment for documents (spec section 8.1).

Rules, applied per document with a fixed seed so results reproduce:

- public → ``{Everyone}`` (everyone sees public)
- internal → ``{Everyone}`` plus one team role (HR or Finance, by hash)
- confidential → exactly one team role (HR or Finance, by hash), never
  Everyone — the explicit-grant rule
- FUNSD images → ``{Everyone}`` (forms are public test fixtures)

``assign`` is pure (doc fields → roles); ``seed_acls`` writes to Postgres.
"""

from __future__ import annotations

import hashlib

SEED = 5


def assign(doc_id: str, label: str, modality: str) -> list[str]:
    """Assign ``allowed_roles`` for one document (deterministic)."""
    if modality == "image":
        return ["Everyone"]
    if label == "public":
        return ["Everyone"]
    digest = hashlib.blake2b(doc_id.encode(), digest_size=1).digest()[0]
    team = "HR" if digest % 2 else "Finance"
    if label == "internal":
        return ["Everyone", team]
    if label == "confidential":
        return [team]
    msg = f"unknown label: {label!r}"
    raise ValueError(msg)


def seed_acls(conn: object) -> dict[str, int]:
    """Write seeded ACLs for every document; returns role-set counts."""
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("SELECT id, label, modality FROM documents")
        rows = cur.fetchall()
        counts: dict[str, int] = {}
        updates: dict[tuple[str, ...], list[str]] = {}
        for doc_id, label, modality in rows:
            roles = assign(doc_id, label, modality)
            key = "+".join(sorted(roles))
            counts[key] = counts.get(key, 0) + 1
            updates.setdefault(tuple(roles), []).append(doc_id)
        for roles, ids in updates.items():
            cur.execute(
                "UPDATE documents SET allowed_roles = %s WHERE id = ANY(%s)",
                (list(roles), ids),
            )
    conn.commit()  # type: ignore[attr-defined]
    return counts
