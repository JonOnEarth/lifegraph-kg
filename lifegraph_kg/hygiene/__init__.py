# SPDX-License-Identifier: Apache-2.0
"""Hygiene engine — canonicalization, dedup, normalization.

Opt-in: ``LifeGraph(hygiene=False)`` skips this module's deps entirely
(notably the embedding model used for fuzzy dedup). Added in L3.
"""
