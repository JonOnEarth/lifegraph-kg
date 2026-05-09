# SPDX-License-Identifier: Apache-2.0
"""Extraction — natural language to typed entities.

Two authoring surfaces (L1):
- ``schema``: Pydantic models (Graphiti-style, type-safe)
- ``examples``: LangExtract-style few-shot ``examples=[...]``

Both converge to the same internal ``ExtractionRequest``.
"""
