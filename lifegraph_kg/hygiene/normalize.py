# SPDX-License-Identifier: Apache-2.0
"""String normalization for hygiene comparisons.

Hygiene answers "are these two entities the same?" via a series of
increasingly fuzzy comparisons. The normalizer provides the canonical
form used by every comparison — a single function that maps surface
strings to a comparable representation.

Normalization steps:
  1. Unicode NFKC normalize  — fullwidth/halfwidth + composed/decomposed
  2. Casefold                  — Unicode-aware lowercase (`İ` → `i̇`, etc.)
  3. Whitespace collapse       — runs of whitespace → single space, strip
  4. Punctuation strip         — drop common trailing punctuation
                                 (Chinese 。，、 + Latin .,;:!?)

The result is suitable for equality comparison and Levenshtein
distance. It is NOT a hash and not stable across Unicode-version
upgrades — only use for comparison, never as a persistent key.

Role-term canonicalization (kinship / role labels) is a separate
concern that belongs in the dedup engine — many "the boss" / "老板"
mentions refer to the *same* person across episodes, and that needs
LLM signal beyond string normalization.
"""

from __future__ import annotations

import re
import unicodedata

_TRAILING_PUNCT = ".,;:!?。，、"
_WHITESPACE = re.compile(r"\s+")


def canonical_form(s: str) -> str:
    """Return the canonical comparable form of a surface string.

    >>> canonical_form("  Sara  ")
    'sara'
    >>> canonical_form("Sara.")
    'sara'
    >>> canonical_form("İstanbul")
    'i̇stanbul'
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _WHITESPACE.sub(" ", s).strip()
    while s and s[-1] in _TRAILING_PUNCT:
        s = s[:-1].rstrip()
    return s


def normalized_value_eq(a: str, b: str) -> bool:
    """Convenience: are two surface strings equal under canonicalization?"""
    return canonical_form(a) == canonical_form(b)
