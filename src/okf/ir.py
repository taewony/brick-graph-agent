#!/usr/bin/env python3
"""
OKF Intermediate Representation (IR) Module
-------------------------------------------
Defines the graph node structures and in-memory representation of OKF bundles.
Used as the compilation target before applying history transformation operators.
"""

from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OKFNode:
    id: str
    title: str
    type: str  # AtomicConcept, CompositeConcept, Module, etc.
    status: str
    path: str
    bundle: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    body: str = ""
    prerequisites: Set[str] = field(default_factory=set)
    composed_of: Set[str] = field(default_factory=set)
    contradicts: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OKFIR:
    def __init__(self):
        self.nodes: Dict[str, OKFNode] = {}
        self.history_applied: List[Dict[str, Any]] = []

    def add_node(self, node: OKFNode):
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[OKFNode]:
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]

    def to_dict() -> Dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "nodes": {
                node_id: {
                    "id": n.id,
                    "title": n.title,
                    "type": n.type,
                    "status": n.status,
                    "path": n.path,
                    "bundle": n.bundle,
                    "prerequisites": sorted(list(n.prerequisites)),
                    "composed_of": sorted(list(n.composed_of)),
                    "contradicts": sorted(list(n.contradicts)),
                    "metadata": n.metadata
                }
                for node_id, n in self.nodes.items()
            }
        }
