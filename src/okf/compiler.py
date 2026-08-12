#!/usr/bin/env python3
"""
OKF Dynamic Graph Compiler Module
----------------------------------
Main entry point for compiling OKF markdown bundles into an executable graph and behaviors.yaml.

Pipeline Steps:
1. Embedded Validation: Runs OKFValidator to verify frontmatter, link integrity, and DAG properties.
2. IR Ingestion: Converts markdown nodes & metadata into OKFIR.
3. History Transformation: Ingests `00_agent_model/history.yaml` and applies SPLIT/MERGE/REORDER operators.
4. ActiveGraph Behaviors Generation: Emits `00_agent_model/behaviors.yaml` (Final State).
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

try:
    from .validator import OKFValidator, ValidationReport
    from .ir import OKFIR, OKFNode
    from .history import OKFHistoryProcessor
except ImportError:
    from validator import OKFValidator, ValidationReport
    from ir import OKFIR, OKFNode
    from history import OKFHistoryProcessor


class OKFCompiler:
    def __init__(self, okf_root: Path):
        self.root = okf_root.resolve()
        self.agent_model_dir = self.root / "00_agent_model"
        self.history_file = self.agent_model_dir / "history.yaml"
        self.output_behaviors_file = self.agent_model_dir / "behaviors.yaml"

    def compile(self, strict_validation: bool = False, verbose: bool = True) -> bool:
        print("\n" + "=" * 60)
        print("🛠️  Executing OKF Compiler Subsystem")
        print("=" * 60)

        # Step 1: Embedded KB Validation
        print("\n[Step 1/4] Embedded KB Validation & Integrity Check...")
        validator = OKFValidator(self.root)
        report = validator.run_validation(verbose=False)
        report.print_summary(verbose=False)

        if not report.is_valid and strict_validation:
            print("❌ Compilation aborted due to KB validation errors (strict_validation=True).")
            return False

        # Step 2: Build In-Memory IR
        print("\n[Step 2/4] Building Intermediate Representation (IR)...")
        ir = OKFIR()

        for file_path, data in validator.files_data.items():
            rel_path_str = str(data['rel_path']).replace('\\', '/')
            bundle_name = rel_path_str.split('/')[0] if '/' in rel_path_str else 'root'
            fm = data['frontmatter']

            # Extract prerequisites & composed_of lists
            prereqs = set()
            val_p = fm.get('prerequisites')
            if isinstance(val_p, list):
                prereqs.update(val_p)
            elif isinstance(val_p, str):
                prereqs.add(val_p)

            composed = set()
            val_c = fm.get('composed_of')
            if isinstance(val_c, list):
                composed.update(val_c)
            elif isinstance(val_c, str):
                composed.add(val_c)

            node = OKFNode(
                id=data['id'],
                title=fm.get('title', file_path.stem),
                type=fm.get('type', 'Concept'),
                status=fm.get('status', 'draft'),
                path=rel_path_str,
                bundle=bundle_name,
                frontmatter=fm,
                body=data['body'],
                prerequisites=prereqs,
                composed_of=composed
            )
            ir.add_node(node)

        print(f"  - Total IR Nodes ingested: {len(ir.nodes)}")

        # Step 3: History Transformation Operators
        print("\n[Step 3/4] Ingesting history.yaml & applying transformation operators...")
        history_processor = OKFHistoryProcessor(self.history_file)
        history_logs = history_processor.apply_to_ir(ir)

        for log in history_logs:
            print(f"  - {log}")
        if not history_logs:
            print("  - No transformation operators found in history.yaml (baseline IR maintained).")

        # Step 4: Emit Final Behaviors & Executable Control Graph Schema (behaviors.yaml)
        print("\n[Step 4/4] Generating final state specification (behaviors.yaml)...")
        behaviors_spec = self._build_behaviors_spec(ir, report)

        self.agent_model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_behaviors_file, 'w', encoding='utf-8') as f:
            yaml.dump(behaviors_spec, f, allow_unicode=True, sort_keys=False)

        print(f"  - Output successfully written to: {self.output_behaviors_file}")
        print("\n🎉 OKF Compilation completed successfully!")
        print("=" * 60 + "\n")
        return True

    def _build_behaviors_spec(self, ir: OKFIR, report: ValidationReport) -> Dict[str, Any]:
        nodes_by_bundle: Dict[str, List[Dict[str, Any]]] = {}
        active_behaviors: List[Dict[str, Any]] = []

        for node_id, node in ir.nodes.items():
            bundle = node.bundle
            if bundle not in nodes_by_bundle:
                nodes_by_bundle[bundle] = []

            nodes_by_bundle[bundle].append({
                "id": node.id,
                "title": node.title,
                "type": node.type,
                "status": node.status,
                "path": node.path,
                "prerequisites": sorted(list(node.prerequisites)),
                "composed_of": sorted(list(node.composed_of))
            })

            # Derive ActiveGraph Cypher behaviors for composite concepts or agent models
            if node.type in ("CompositeConcept", "Module") or node.bundle == "00_agent_model":
                cypher_pattern = f"""MATCH (c:Concept {{id: '{node.id}'}})
WHERE all(p IN [(p:Concept)-[:PREREQUISITE_OF]->(c) | p] WHERE p.status = 'mastered')
RETURN c"""
                active_behaviors.append({
                    "behavior_id": f"behavior.{node.id.replace('.', '_')}",
                    "target_node": node.id,
                    "trigger_event": "COMPOSITE_ASSEMBLED" if node.type == "CompositeConcept" else "GRAPH_COMPILED",
                    "cypher_pattern": cypher_pattern,
                    "handler": f"handle_{node.id.replace('.', '_')}_assembled"
                })

        behaviors_spec = {
            "version": "1.10.0",
            "compiler": "OKFCompiler/2.0",
            "compiled_at": "2026-08-12T17:15:00Z",
            "summary": {
                "total_nodes": len(ir.nodes),
                "total_bundles": len(nodes_by_bundle),
                "history_operators_applied": len(ir.history_applied),
                "active_behaviors_count": len(active_behaviors)
            },
            "history_applied": ir.history_applied,
            "behaviors": active_behaviors,
            "graph_nodes_by_bundle": nodes_by_bundle
        }
        return behaviors_spec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OKF Dynamic Graph Compiler")
    parser.add_argument("path", nargs="?", default="D:/code/brick-graph-agent/.okf", help="Path to .okf directory")
    parser.add_argument("--strict", action="store_true", help="Halt on validation errors")
    args = parser.parse_args()

    compiler = OKFCompiler(Path(args.path))
    success = compiler.compile(strict_validation=args.strict, verbose=True)
    sys.exit(0 if success else 1)
