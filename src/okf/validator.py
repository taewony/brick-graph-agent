#!/usr/bin/env python3
"""
OKF KB Validator Module
------------------------
Validates OKF (Open Knowledge Format) markdown bundles across:
1. Frontmatter syntax & schema validation
2. Node ID collision & mapping
3. Link integrity (Wiki links, Markdown links, file:// links, relationship blocks)
4. Dangling references & unresolvable target IDs
5. Circular dependency detection (PREREQUISITES / COMPOSED_OF cycles)
"""

import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')



@dataclass
class ValidationError:
    file_path: Path
    level: str  # 'ERROR' or 'WARNING'
    code: str   # 'ID_COLLISION', 'BROKEN_LINK', 'MISSING_FIELD', 'CIRCULAR_DEP', 'INVALID_YAML'
    message: str
    target: Optional[str] = None


@dataclass
class ValidationReport:
    is_valid: bool
    total_files: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    id_map: Dict[str, Path] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def print_summary(self, verbose: bool = False):
        print("\n" + "=" * 60)
        print("🔍 OKF KB Validation Report")
        print("=" * 60)
        print(f"📁 Total Files Scanned : {self.total_files}")
        print(f"❌ Total Errors       : {len(self.errors)}")
        print(f"⚠️  Total Warnings     : {len(self.warnings)}")
        print(f" STATUS               : {'✅ PASS' if self.is_valid else '❌ FAIL'}")
        print("-" * 60)

        if self.errors:
            print("\n🚨 ERRORS:")
            for err in self.errors:
                print(f"  - [{err.code}] {err.file_path}: {err.message}")

        if self.warnings and (verbose or not self.errors):
            print("\n⚠️ WARNINGS:")
            for warn in self.warnings:
                print(f"  - [{warn.code}] {warn.file_path}: {warn.message}")
        print("=" * 60 + "\n")


class OKFValidator:
    def __init__(self, okf_root: Path):
        self.root = okf_root.resolve()
        self.id_map: Dict[str, Path] = {}
        self.files_data: Dict[Path, Dict[str, Any]] = {}

    def collect_md_files() -> List[Path]:
        pass

    def run_validation(self, verbose: bool = False) -> ValidationReport:
        md_files = list(self.root.rglob("*.md"))
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # Step 1: Parse all files & build ID Map
        self.id_map = {}
        self.files_data = {}

        for file_path in md_files:
            rel_path = file_path.relative_to(self.root)
            content = file_path.read_text(encoding='utf-8')
            frontmatter, body, yaml_err = self._parse_frontmatter(content, file_path)

            if yaml_err:
                errors.append(ValidationError(
                    file_path=rel_path,
                    level='ERROR',
                    code='INVALID_YAML',
                    message=f"YAML parsing error: {yaml_err}"
                ))

            node_id = frontmatter.get('id')
            if not node_id:
                # Scope index and log files by relative path to prevent collisions
                if file_path.stem in ('index', 'log', 'README'):
                    node_id = str(rel_path.with_suffix('')).replace('\\', '/')
                else:
                    node_id = file_path.stem
                if 'id' not in frontmatter and not file_path.name.startswith(("index", "log")):
                    warnings.append(ValidationError(
                        file_path=rel_path,
                        level='WARNING',
                        code='MISSING_FIELD',
                        message=f"Missing explicit 'id' in frontmatter, defaulting to '{node_id}'"
                    ))

            # ID collision check
            if node_id in self.id_map:
                errors.append(ValidationError(
                    file_path=rel_path,
                    level='ERROR',
                    code='ID_COLLISION',
                    message=f"Duplicate Node ID '{node_id}' (Conflict with {self.id_map[node_id].relative_to(self.root)})"
                ))
            else:
                self.id_map[node_id] = file_path

            # Also register stem as secondary fallback
            stem = file_path.stem
            if stem != node_id and stem not in self.id_map:
                self.id_map[stem] = file_path

            self.files_data[file_path] = {
                'id': node_id,
                'frontmatter': frontmatter,
                'body': body,
                'rel_path': rel_path
            }

        # Step 2: Validate Links and Relationship Targets
        for file_path, data in self.files_data.items():
            rel_path = data['rel_path']
            body = data['body']
            fm = data['frontmatter']

            # Extract targets
            targets = self._extract_all_targets(body, fm)

            for target, target_type in targets:
                resolved_ok, resolved_path, reason = self._resolve_target(target, file_path)
                if not resolved_ok:
                    if target in self.id_map:
                        continue  # Valid ID lookup
                    # Check if target is a known relationship or stem ID
                    if target_type == 'relationship':
                        errors.append(ValidationError(
                            file_path=rel_path,
                            level='ERROR',
                            code='DANGLING_RELATIONSHIP',
                            message=f"Unresolved relationship target '{target}'",
                            target=target
                        ))
                    else:
                        warnings.append(ValidationError(
                            file_path=rel_path,
                            level='WARNING',
                            code='BROKEN_LINK',
                            message=f"Unresolvable link '{target}' ({reason})",
                            target=target
                        ))

        # Step 3: Circular Dependency Detection
        cycles = self._detect_circular_dependencies()
        for cycle in cycles:
            errors.append(ValidationError(
                file_path=Path(cycle[0]),
                level='ERROR',
                code='CIRCULAR_DEP',
                message=f"Circular dependency detected: {' -> '.join(cycle)}"
            ))

        is_valid = len(errors) == 0
        report = ValidationReport(
            is_valid=is_valid,
            total_files=len(md_files),
            errors=errors,
            warnings=warnings,
            id_map=self.id_map,
            stats={
                'total_nodes': len(self.id_map),
                'total_files': len(md_files)
            }
        )
        return report

    def _parse_frontmatter(self, content: str, file_path: Path) -> Tuple[Dict, str, Optional[str]]:
        if not content.startswith('---'):
            return {}, content, None
        parts = content.split('---', 2)
        if len(parts) < 3:
            return {}, content, None
        try:
            fm = yaml.safe_load(parts[1]) or {}
            return fm, parts[2].strip(), None
        except Exception as e:
            return {}, content, str(e)

    def _extract_all_targets(self, body: str, fm: Dict) -> List[Tuple[str, str]]:
        targets: List[Tuple[str, str]] = []

        # Wiki links [[target]]
        for match in re.findall(r'\[\[([^\]]+)\]\]', body):
            targets.append((match.strip(), 'wiki'))

        # Markdown links [text](path)
        for _, path in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body):
            if not path.startswith(('http://', 'https://', '#')):
                targets.append((path.strip(), 'markdown'))

        # Frontmatter relationship fields
        rel_fields = ['prerequisites', 'composed_of', 'contradicts', 'references', 'composes']
        for field in rel_fields:
            val = fm.get(field)
            items = val if isinstance(val, list) else ([val] if val else [])
            for item in items:
                if isinstance(item, str):
                    # Clean trailing parenthetical explanations e.g. "target (Module 04)"
                    clean_item = re.sub(r'\s*\([^)]*\)', '', item).strip()
                    if clean_item:
                        targets.append((clean_item, 'relationship'))

        return targets

    def _resolve_target(self, target: str, current_file: Path) -> Tuple[bool, Optional[Path], str]:
        if not target:
            return False, None, "Empty target"

        # Direct ID map match
        if target in self.id_map:
            return True, self.id_map[target], ""

        # file:/// link
        if target.startswith('file:///'):
            path_str = target.replace('file:///', '')
            if path_str.startswith('/') and ':' in path_str[1:]:
                path_str = path_str[1:]
            try:
                p = Path(path_str).resolve()
                if p.exists():
                    return True, p, ""
            except Exception:
                pass

        # Relative path resolution
        clean_target = target.split('#')[0].split('?')[0]
        base_dir = current_file.parent
        candidate = (base_dir / clean_target).resolve()
        if candidate.exists():
            return True, candidate, ""

        if not clean_target.endswith('.md'):
            candidate_md = (base_dir / (clean_target + '.md')).resolve()
            if candidate_md.exists():
                return True, candidate_md, ""

        # Root relative candidate
        candidate_root = (self.root / clean_target).resolve()
        if candidate_root.exists():
            return True, candidate_root, ""

        if not clean_target.endswith('.md'):
            candidate_root_md = (self.root / (clean_target + '.md')).resolve()
            if candidate_root_md.exists():
                return True, candidate_root_md, ""

        # Stem name lookup
        stem = Path(clean_target).stem
        if stem in self.id_map:
            return True, self.id_map[stem], ""

        return False, None, f"Could not locate file or node for '{target}'"

    def _detect_circular_dependencies(self) -> List[List[str]]:
        graph: Dict[str, Set[str]] = {}
        for data in self.files_data.values():
            node_id = data['id']
            fm = data['frontmatter']
            deps = set()
            for field in ['prerequisites', 'composed_of']:
                val = fm.get(field)
                if isinstance(val, list):
                    deps.update(val)
                elif isinstance(val, str):
                    deps.add(val)
            graph[node_id] = deps

        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            rec_stack.remove(node)
            path.pop()

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles


if __name__ == "__main__":
    import sys
    okf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/code/brick-graph-agent/.okf")
    validator = OKFValidator(okf_path)
    report = validator.run_validation(verbose=True)
    report.print_summary()
    sys.exit(0 if report.is_valid else 1)
