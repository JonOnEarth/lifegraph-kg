# SPDX-License-Identifier: Apache-2.0
"""LLM provider abstraction.

A single ``LlmClient`` interface (L1) with one default implementation
(Anthropic). Per the v0.1 design, the user supplies their own API key;
the library never owns LLM credentials.
"""
