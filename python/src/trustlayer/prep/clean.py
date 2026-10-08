"""Email text cleaning. Pure Python, no pyspark."""

from __future__ import annotations

import hashlib
import re

_REPLY_PREFIXES = ("re:", "fw:", "fwd:")


def normalize_subject(subject: str) -> str:
    """Lowercase a subject line and strip reply/forward prefixes."""
    text = subject.strip().lower()
    while True:
        for prefix in _REPLY_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        else:
            return text


def thread_id_from_subject(subject: str) -> str:
    """Derive a stable pseudo-thread id from a subject line."""
    digest = hashlib.sha256(normalize_subject(subject).encode("utf-8")).hexdigest()
    return digest[:12]


def clean_text(body: str) -> str:
    """Strip quoted replies, forwarded headers, and signatures from a body."""
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    kept = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if _is_signature_marker(stripped):
            break
        if _is_quoted(stripped) or _is_forwarded_header(stripped):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _is_quoted(stripped: str) -> bool:
    """Check for quote markers left by reply chains."""
    return stripped.startswith((">", "|"))


def _is_forwarded_header(stripped: str) -> bool:
    """Check for forwarded-message separators and attribution lines."""
    return stripped.startswith("-----") or (
        stripped.startswith("On ") and stripped.endswith("wrote:")
    )


def _is_signature_marker(stripped: str) -> bool:
    """Check for the start of a signature block; everything after is dropped."""
    return stripped in ("--", "---") or stripped.lower().startswith("sent from my")
