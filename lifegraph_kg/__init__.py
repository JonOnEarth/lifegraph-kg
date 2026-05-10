# SPDX-License-Identifier: Apache-2.0
"""lifegraph-kg — autobiographical memory engine.

Public surface (stable from v0.1):

    from lifegraph_kg import LifeGraph, classes
    from lifegraph_kg.classes import Person, Place, Project, Topic
    from lifegraph_kg.extract import extract
    from lifegraph_kg.extract.schema import ExtractionResult

The 4 default life-classes (Person/Place/Project/Topic) are anchored to
the canonical PKG ontology — PIMO, Balog & Kenter, Conway's Self-Memory
System, DOLCE, CIDOC-CRM. See `lifegraph_kg.classes` for the lineage.
"""

from importlib.metadata import PackageNotFoundError, version

from lifegraph_kg import classes
from lifegraph_kg.classes import Person, Place, Project, Topic
from lifegraph_kg.extract import extract
from lifegraph_kg.extract.schema import ExtractionResult
from lifegraph_kg.kg import LifeGraph

try:
    __version__ = version("lifegraph-kg")
except PackageNotFoundError:
    __version__ = "0.0.1.dev0"


__all__ = [
    "ExtractionResult",
    "LifeGraph",
    "Person",
    "Place",
    "Project",
    "Topic",
    "__version__",
    "classes",
    "extract",
]
