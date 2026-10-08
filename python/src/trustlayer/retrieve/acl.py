"""Access-control predicate: who may see which document.

Spec section 8.1: users alice (HR), bob (Finance), carol (Everyone),
admin. A user may see a document if one of their roles is in the
document's ``allowed_roles``; confidential documents additionally
require an explicit role grant (never inherited from Everyone).

Pure Python (no psycopg): unit-tested without a database. The SQL
pre-filter in ``search.py`` must implement exactly this predicate —
defence in depth re-checks every returned row against it.
"""

from __future__ import annotations

USERS: dict[str, tuple[str, ...]] = {
    "alice": ("HR", "Everyone"),
    "bob": ("Finance", "Everyone"),
    "carol": ("Everyone",),
    "admin": ("HR", "Finance", "Everyone", "Admin"),
}

LABEL_RANK = {"public": 0, "internal": 1, "confidential": 2}


def visible(label: str, allowed_roles: list[str], user_roles: list[str]) -> bool:
    """Decide whether a user with ``user_roles`` may see the document.

    Confidential needs an explicit (non-Everyone) role in common with
    the grant; other labels need any role in common.
    """
    shared = set(allowed_roles) & set(user_roles)
    if not shared:
        return False
    if label == "confidential":
        return bool(shared - {"Everyone"})
    return True
