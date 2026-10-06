"""Exact-repeat detection for teacher replies."""

from __future__ import annotations


def normalize_exact_teacher_output(text: str | None) -> str:
    """Canonical text for exact-repeat detection.

    Only whitespace is normalized. Case, punctuation, words, numbers, and math
    remain significant so a stable teaching scaffold with new content is not
    mistaken for a repeat.
    """
    return " ".join((text or "").split())
