# SPDX-License-Identifier: Apache-2.0
"""External benchmark adapters.

Each adapter translates an external benchmark's input format into
``lg.log(...)`` calls and the benchmark's expected-output format into one
of our scoring categories. Running these requires the external benchmark
data (often gated by license — see each adapter's README) and frequently
running competitor libraries side-by-side.

Phase target: L5 (post-v0.1 launch).
"""
