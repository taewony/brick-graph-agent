#!/usr/bin/env python3
"""
OKF History Operator Processor
------------------------------
Reads `00_agent_model/history.yaml` and applies structural transformation operators
(SPLIT, MERGE, REORDER, RENAME, UPDATE_STATUS, UPDATE_PREREQUISITE) onto the OKF IR.
Ensures that the compiled `behaviors.yaml` reflects the final state after all historical operations.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from .ir import OKFIR, OKFNode


class OKFHistoryProcessor:
    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = history_file
        self.operations: List[Dict[str, Any]] = []
        if history_file and history_file.exists():
            self._load_history(history_file)

    def _load_history(self, path: Path):
        try:
            content = path.read_text(encoding='utf-8')
            data = yaml.safe_load(content) or {}
            if isinstance(data, list):
                self.operations = data
            elif isinstance(data, dict):
                self.operations = data.get('history', [])
        except Exception as e:
            print(f"⚠️ Warning loading history log from {path}: {e}")

    def apply_to_ir(self, ir: OKFIR) -> List[str]:
        logs: List[str] = []
        for index, op in enumerate(self.operations):
            op_type = op.get('op') or op.get('type')
            if not op_type:
                continue
            op_type = str(op_type).upper()

            if op_type == 'SPLIT':
                self._apply_split(ir, op, logs)
            elif op_type == 'MERGE':
                self._apply_merge(ir, op, logs)
            elif op_type == 'RENAME':
                self._apply_rename(ir, op, logs)
            elif op_type in ('UPDATE_PREREQUISITE', 'REORDER'):
                self._apply_update_prerequisite(ir, op, logs)
            elif op_type == 'UPDATE_STATUS':
                self._apply_update_status(ir, op, logs)
            else:
                logs.append(f"⚠️ Unknown history operator '{op_type}' at step {index}")

        ir.history_applied = self.operations
        return logs

    def _apply_split(self, ir: OKFIR, op: Dict[str, Any], logs: List[str]):
        target_id = op.get('target')
        new_node_ids = op.get('new_nodes', [])
        deprecate_target = op.get('deprecate_target', True)

        target_node = ir.get_node(target_id)
        if target_node and deprecate_target:
            target_node.status = 'deprecated'
            target_node.metadata['deprecated_by'] = new_node_ids
            logs.append(f"✂️ SPLIT: Node '{target_id}' deprecated in favor of {new_node_ids}")

    def _apply_merge(self, ir: OKFIR, op: Dict[str, Any], logs: List[str]):
        sources = op.get('sources', [])
        target_id = op.get('target')

        target_node = ir.get_node(target_id)
        for src_id in sources:
            src_node = ir.get_node(src_id)
            if src_node:
                src_node.status = 'merged'
                src_node.metadata['merged_into'] = target_id
                if target_node:
                    target_node.prerequisites.update(src_node.prerequisites)
                    target_node.composed_of.update(src_node.composed_of)
        logs.append(f"🔀 MERGE: Merged {sources} into '{target_id}'")

    def _apply_rename(self, ir: OKFIR, op: Dict[str, Any], logs: List[str]):
        old_id = op.get('old_id')
        new_id = op.get('new_id')

        if old_id and new_id and old_id in ir.nodes:
            node = ir.nodes.pop(old_id)
            node.id = new_id
            ir.nodes[new_id] = node

            # Update relationship references
            for n in ir.nodes.values():
                if old_id in n.prerequisites:
                    n.prerequisites.remove(old_id)
                    n.prerequisites.add(new_id)
                if old_id in n.composed_of:
                    n.composed_of.remove(old_id)
                    n.composed_of.add(new_id)
            logs.append(f"🏷️ RENAME: Renamed '{old_id}' -> '{new_id}'")

    def _apply_update_prerequisite(self, ir: OKFIR, op: Dict[str, Any], logs: List[str]):
        target_id = op.get('target')
        prereqs = op.get('prerequisites', [])
        node = ir.get_node(target_id)
        if node:
            node.prerequisites = set(prereqs)
            logs.append(f"🔄 REORDER: Updated prerequisites for '{target_id}' to {prereqs}")

    def _apply_update_status(self, ir: OKFIR, op: Dict[str, Any], logs: List[str]):
        target_id = op.get('target')
        new_status = op.get('status', 'mastered')
        node = ir.get_node(target_id)
        if node:
            node.status = new_status
            logs.append(f"✅ UPDATE_STATUS: Updated '{target_id}' status to '{new_status}'")
