"""Rule-based weak labeller. Pure Python, no pyspark.

Label scheme (spec section 8.1, roadmap Phase 3):

    Label.PUBLIC        no PII flags, no sensitivity keywords
    Label.INTERNAL      PII flags but low-severity (name, email), or internal
                        business keywords (meeting, budget, report)
    Label.CONFIDENTIAL  high-severity PII (SSN, card, passport, credentials)
                        or explicit confidentiality markers

PII flags are coarse by design: the ai4privacy dataset (SOURCES.md) provides
 genuine span-level labels for Phase 7 training; these rules exist to
weak-label the email corpus cheaply and to stratify the gold sample.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PASSPORT_RE = re.compile(
    r"(?i)\bpassport\s*(?:no|number|#)?\s*[:.]?\s*[A-Z0-9]{6,12}\b"
)
_CRED_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd|api[_-]?key|secret)\b\s*[:=]\s*\S+"
)
_ACCOUNT_RE = re.compile(
    r"(?i)\b(?:account|routing|iban|swift)\b\s*(?:no|number|#)?\s*[:.]?\s*[\dA-Z-]{6,34}\b"
)
_CONFIDENTIAL_RE = re.compile(
    r"(?i)\b(?:confidential|privileged|attorney|do not (?:forward|distribute)|"
    r"sensitive|private and confidential)\b"
)
_INTERNAL_RE = re.compile(
    r"(?i)\b(?:meeting|budget|forecast|headcount|salary|salaries|layoff|merger|acquisition|"
    r"report|invoice|contract|deadline|quarterly|strategy|internal only)\b"
)

_HIGH_SEVERITY = frozenset({"ssn", "card", "passport", "credentials", "account"})


class Label(Enum):
    """Sensitivity label for a chunk."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


@dataclass(frozen=True)
class WeakLabel:
    """Weak label plus the evidence that produced it."""

    label: Label
    pii: tuple[str, ...]
    rules: tuple[str, ...]


def weak_label(title: str, text: str) -> WeakLabel:
    """Label one chunk from subject + body; rules fire on either field."""
    blob = f"{title}\n{text}"
    pii: list[str] = []
    rules: list[str] = []

    def flag(name: str, pattern: re.Pattern[str], rule: str) -> None:
        if pattern.search(blob):
            pii.append(name)
            rules.append(rule)

    flag("email", _EMAIL_RE, "email-address")
    flag("phone", _PHONE_RE, "phone-number")
    flag("ssn", _SSN_RE, "ssn")
    flag("passport", _PASSPORT_RE, "passport")
    flag("credentials", _CRED_RE, "credentials")
    flag("account", _ACCOUNT_RE, "account-number")
    if _CARD_RE.search(blob) and _has_luhn(blob):
        pii.append("card")
        rules.append("card-luhn")

    confidential = _CONFIDENTIAL_RE.search(blob) is not None
    if confidential:
        rules.append("confidential-marker")
    internal_kw = _INTERNAL_RE.search(blob) is not None
    if internal_kw:
        rules.append("internal-keyword")

    if confidential or _HIGH_SEVERITY.intersection(pii):
        label = Label.CONFIDENTIAL
    elif pii or internal_kw:
        label = Label.INTERNAL
    else:
        label = Label.PUBLIC
    return WeakLabel(label, tuple(pii), tuple(rules))


def _has_luhn(blob: str) -> bool:
    """Check whether any 13–19 digit run passes the Luhn checksum."""
    for match in _CARD_RE.finditer(blob):
        digits = [c for c in match.group(0) if c.isdigit()]
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


def _luhn_ok(digits: list[str]) -> bool:
    """Validate a digit list with the Luhn mod-10 checksum."""
    total = 0
    for i, char in enumerate(reversed(digits)):
        num = int(char)
        if i % 2 == 1:
            num *= 2
            if num > 9:
                num -= 9
        total += num
    return total % 10 == 0
