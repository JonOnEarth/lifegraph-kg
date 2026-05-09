# SPDX-License-Identifier: Apache-2.0
"""lifegraph-kg — autobiographical memory engine.

Public surface (stable from v0.1; pre-alpha until then):

    from lifegraph_kg import LifeGraph, classes

The default life schema lives in ``lifegraph_kg.classes``; the knowledge-graph
facade and storage drivers live in ``lifegraph_kg.kg``; the extractor in
``lifegraph_kg.extract``; the opt-in hygiene engine in ``lifegraph_kg.hygiene``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lifegraph-kg")
except PackageNotFoundError:
    __version__ = "0.0.1.dev0"

# Re-exports land here once the modules exist (L1+):
#   from lifegraph_kg.kg import LifeGraph
#   from lifegraph_kg import classes
__all__ = ["__version__"]
