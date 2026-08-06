#!/usr/bin/env python3
r"""
OKF KB Link Checker - Recursive Module Dependency Checker with Verbose Mode
----------------------------------------------------------------------------
- module 파일에서 시작하여 composite, atomic까지 재귀적으로 모든 링크를 검사합니다.
- 자기 자신을 가리키는 링크는 출력하지 않습니다.
- -v 옵션으로 유효한 링크까지 모두 출력합니다 (중복 제거, 간결한 경로).

사용법:
    # 전체 검사
    python link_check.py .

    # Module 1 검사 + 유효한 링크 출력
    python link_check.py -m 1 -v
"""

import os
import re
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# ----------------------------------------------------------------------
# 1. 파일 수집 및 ID 맵 구축
# ----------------------------------------------------------------------

def collect_md_files(root: Path) -> List[Path]:
    return list(root.rglob("*.md"))

def parse_frontmatter(file_path: Path) -> Dict:
    content = file_path.read_text(encoding='utf-8')
    if not content.startswith('---'):
        return {}
    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}

def build_id_map(files: List[Path]) -> Dict[str, Path]:
    id_map = {}
    for f in files:
        fm = parse_frontmatter(f)
        node_id = fm.get('id')
        if not node_id:
            node_id = f.stem
        if node_id in id_map:
            print(f"⚠️ 중복 ID 발견: '{node_id}' in {f} and {id_map[node_id]}")
        id_map[node_id] = f
        stem = f.stem
        if stem != node_id and stem not in id_map:
            id_map[stem] = f
    return id_map

# ----------------------------------------------------------------------
# 2. 링크/ID 추출
# ----------------------------------------------------------------------

def extract_wiki_links(content: str) -> List[str]:
    return re.findall(r'\[\[([^\]]+)\]\]', content)

def extract_markdown_links(content: str) -> List[Tuple[str, str]]:
    return re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

def extract_file_links(content: str) -> List[str]:
    return re.findall(r'file:///[^\s\)]+', content)

def extract_relationships(content: str) -> List[str]:
    pattern = r'(?i)^\s*#{2,3}\s*relationships\s*$'
    lines = content.splitlines()
    in_section = False
    ids = []
    for line in lines:
        if re.match(pattern, line.strip()):
            in_section = True
            continue
        if in_section:
            if re.match(r'^\s*#{1,6}\s+', line):
                break
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    value = parts[1].strip()
                    for token in re.split(r'[,;]\s*', value):
                        token = token.strip()
                        if token:
                            ids.append(token)
            elif line.strip().startswith('-'):
                item = line.strip()[1:].strip()
                if ':' in item:
                    parts = item.split(':', 1)
                    if len(parts) == 2:
                        value = parts[1].strip()
                        for token in re.split(r'[,;]\s*', value):
                            token = token.strip()
                            if token:
                                ids.append(token)
                else:
                    ids.append(item)
    return ids

def extract_all_targets(content: str) -> List[str]:
    targets = []
    targets.extend(extract_wiki_links(content))
    for _, path in extract_markdown_links(content):
        if not path.startswith(('http://', 'https://', '#')):
            targets.append(path)
    targets.extend(extract_file_links(content))
    targets.extend(extract_relationships(content))
    return targets

# ----------------------------------------------------------------------
# 3. 링크 검증 및 정규화
# ----------------------------------------------------------------------

def normalize_target_to_path(target: str, current_file: Path, root: Path) -> Tuple[bool, Optional[Path], str]:
    original_target = target

    if target.startswith('file:///'):
        path_str = target.replace('file:///', '')
        if path_str.startswith('/') and ':' in path_str[1:]:
            path_str = path_str[1:]
        try:
            p = Path(path_str).resolve()
        except Exception:
            return False, None, f"Invalid path: {original_target}"
        if p == current_file.resolve():
            return False, None, "Self-reference"
        return True, p, ""

    if '#' in target:
        target = target.split('#')[0]
    if '?' in target:
        target = target.split('?')[0]
    if not target.strip():
        return False, None, "Empty path"

    base_dir = current_file.parent
    candidate = (base_dir / target).resolve()
    if candidate.exists():
        if candidate == current_file.resolve():
            return False, None, "Self-reference"
        return True, candidate, ""

    if not target.endswith('.md'):
        candidate_md = (base_dir / (target + '.md')).resolve()
        if candidate_md.exists():
            if candidate_md == current_file.resolve():
                return False, None, "Self-reference"
            return True, candidate_md, ""

    candidate_root = (root / target).resolve()
    if candidate_root.exists():
        if candidate_root == current_file.resolve():
            return False, None, "Self-reference"
        return True, candidate_root, ""

    if not target.endswith('.md'):
        candidate_root_md = (root / (target + '.md')).resolve()
        if candidate_root_md.exists():
            if candidate_root_md == current_file.resolve():
                return False, None, "Self-reference"
            return True, candidate_root_md, ""

    return False, None, f"File not found: {original_target}"

# ----------------------------------------------------------------------
# 4. 모듈 필터링 및 재귀적 파일 확장
# ----------------------------------------------------------------------

def filter_files_by_module(files: List[Path], module_num: int) -> List[Path]:
    prefix = f"module_{module_num:02d}_"
    return [f for f in files if f.stem.startswith(prefix)]

def expand_files_recursively(start_files: List[Path], root: Path, id_map: Dict[str, Path]) -> Set[Path]:
    to_process = set(start_files)
    processed = set()
    all_files = set(start_files)

    while to_process:
        current = to_process.pop()
        if current in processed:
            continue
        processed.add(current)

        content = current.read_text(encoding='utf-8')
        targets = extract_all_targets(content)

        for target in targets:
            ok, path, _ = normalize_target_to_path(target, current, root)
            if ok and path and path.suffix == '.md' and path.is_file():
                if str(path).startswith(str(root)):
                    if path not in processed and path not in to_process:
                        to_process.add(path)
                        all_files.add(path)
            elif target in id_map:
                path = id_map[target]
                if path not in processed and path not in to_process:
                    to_process.add(path)
                    all_files.add(path)
            elif target.startswith('.okf/'):
                candidate = (root / target.replace('.okf/', '')).resolve()
                if candidate.is_file() and candidate not in processed:
                    to_process.add(candidate)
                    all_files.add(candidate)
    return all_files

# ----------------------------------------------------------------------
# 5. 경로 간소화 헬퍼
# ----------------------------------------------------------------------

def simplify_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return str(rel).replace('\\', '/')
    except ValueError:
        return str(path)

# ----------------------------------------------------------------------
# 6. 메인 검사 함수 (재귀적 + 중복 제거)
# ----------------------------------------------------------------------

def check_module(module_num: int, root: Optional[Path] = None, verbose: bool = False):
    if root is None:
        candidates = [Path.cwd(), Path.cwd() / '.okf' / '00_nano_vllm']
        for c in candidates:
            if c.exists() and c.is_dir():
                root = c
                break
        if root is None:
            print("❌ nano-vllm KB 디렉토리를 찾을 수 없습니다.")
            sys.exit(1)
    else:
        root = root.resolve()

    if not root.is_dir():
        print(f"❌ 오류: {root} 는 디렉터리가 아닙니다.")
        sys.exit(1)

    all_files = collect_md_files(root)
    id_map = build_id_map(all_files)

    module_files = filter_files_by_module(all_files, module_num)
    if not module_files:
        print(f"⚠️ Module {module_num}에 해당하는 파일을 찾을 수 없습니다.")
        return

    print(f"🔍 Module {module_num} 검사 시작 (Recursive)")
    expanded_files = expand_files_recursively(module_files, root, id_map)
    sorted_files = sorted(list(expanded_files))

    print(f"📄 관련 파일 총 {len(sorted_files)}개 (재귀적 확장):")
    for f in sorted_files:
        rel = f.relative_to(root)
        print(f"  - {rel}")

    # 중복 제거를 위한 Set
    valid_set = set()       # (source_rel, target_rel)
    broken_links = []       # (source_rel, target, msg)
    external_refs = []      # (source_rel, target, resolved_path)

    for f in sorted_files:
        content = f.read_text(encoding='utf-8')
        rel_path = f.relative_to(root)
        targets = extract_all_targets(content)

        for target in targets:
            if target.startswith(('http://', 'https://')):
                continue

            ok, target_path, msg = normalize_target_to_path(target, f, root)
            target_in_id_map = target in id_map

            if not ok and msg == "Self-reference":
                # 자기 자신은 완전히 무시 (출력 제외)
                continue

            if ok and target_path:
                if not str(target_path).startswith(str(root)):
                    external_refs.append((str(rel_path), target, target_path))
                else:
                    # 중복 제거를 위해 (source, target) 키 생성
                    simplified = simplify_path(target_path, root)
                    valid_set.add((str(rel_path), simplified))
            elif target_in_id_map:
                id_path = id_map[target]
                if id_path != f:
                    if not str(id_path).startswith(str(root)):
                        external_refs.append((str(rel_path), target, id_path))
                    else:
                        simplified = simplify_path(id_path, root)
                        valid_set.add((str(rel_path), simplified))
                else:
                    # 자기 자신은 무시
                    continue
            else:
                broken_links.append((str(rel_path), target, msg))

    # 정렬된 리스트로 변환
    valid_links = sorted(list(valid_set))

    print("\n" + "="*60)
    print(f"📊 Module {module_num} 검사 결과")
    print("="*60)

    if verbose and valid_links:
        print(f"\n📎 유효한 링크 ({len(valid_links)}개):")
        for source, target in valid_links:
            print(f"  - {source}: {target}")

    if broken_links:
        print(f"\n❌ 깨진 링크 ({len(broken_links)}개):")
        for file, target, msg in broken_links:
            print(f"  - {file}: {target}")
            print(f"    이유: {msg}")
    else:
        print("\n✅ 깨진 링크가 없습니다.")

    if external_refs:
        print(f"\n⚠️ 외부 파일 참조 ({len(external_refs)}개):")
        for file, target, resolved_path in external_refs[:5]:
            print(f"  - {file}: {target} -> {resolved_path}")
        if len(external_refs) > 5:
            print(f"  ... 외 {len(external_refs)-5}개")

    print("\n📁 검사된 전체 파일 목록:")
    for f in sorted_files:
        rel = f.relative_to(root)
        print(f"  - {rel}")

def check_links(root: Path, verbose: bool = False):
    root = root.resolve()
    if not root.is_dir():
        print(f"❌ 오류: {root} 는 디렉터리가 아닙니다.")
        sys.exit(1)

    print(f"🔍 전체 검사 시작: {root}")
    md_files = collect_md_files(root)
    id_map = build_id_map(md_files)
    print(f"📄 발견된 .md 파일: {len(md_files)}개")

    valid_set = set()
    broken_links = []
    external_refs = []

    for f in md_files:
        content = f.read_text(encoding='utf-8')
        rel_path = f.relative_to(root)
        targets = extract_all_targets(content)

        for target in targets:
            if target.startswith(('http://', 'https://')):
                continue

            ok, target_path, msg = normalize_target_to_path(target, f, root)
            target_in_id_map = target in id_map

            if not ok and msg == "Self-reference":
                continue

            if ok and target_path:
                if not str(target_path).startswith(str(root)):
                    external_refs.append((str(rel_path), target, target_path))
                else:
                    simplified = simplify_path(target_path, root)
                    valid_set.add((str(rel_path), simplified))
            elif target_in_id_map:
                id_path = id_map[target]
                if id_path != f:
                    if not str(id_path).startswith(str(root)):
                        external_refs.append((str(rel_path), target, id_path))
                    else:
                        simplified = simplify_path(id_path, root)
                        valid_set.add((str(rel_path), simplified))
                else:
                    continue
            else:
                broken_links.append((str(rel_path), target, msg))

    valid_links = sorted(list(valid_set))

    print("\n" + "="*60)
    print("📊 전체 검사 결과")
    print("="*60)

    if verbose and valid_links:
        print(f"\n📎 유효한 링크 ({len(valid_links)}개):")
        for source, target in valid_links:
            print(f"  - {source}: {target}")

    if broken_links:
        print(f"\n❌ 깨진 링크 ({len(broken_links)}개):")
        for file, target, msg in broken_links:
            print(f"  - {file}: {target} -> {msg}")
    else:
        print("\n✅ 깨진 링크가 없습니다.")

    if external_refs:
        print(f"\n⚠️ 외부 파일 참조 ({len(external_refs)}개):")
        for file, target, resolved_path in external_refs[:5]:
            print(f"  - {file}: {target} -> {resolved_path}")
        if len(external_refs) > 5:
            print(f"  ... 외 {len(external_refs)-5}개")

# ----------------------------------------------------------------------
# 7. CLI 진입점
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OKF KB Link Checker - Recursive Module Dependency Checker",
        epilog="예: python link_check.py . -v   또는   python link_check.py -m 3 -v"
    )
    parser.add_argument(
        "path",
        nargs='?',
        default='.',
        help="KB 디렉토리 경로 (기본값: 현재 디렉토리)"
    )
    parser.add_argument(
        "-m", "--module",
        type=int,
        choices=range(0, 8),
        help="검사할 모듈 번호 (0~7) - 관련 파일을 재귀적으로 추적합니다."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="유효한 링크까지 모두 출력합니다 (중복 제거, 간결한 경로)."
    )
    args = parser.parse_args()

    if args.module:
        check_module(args.module, Path(args.path), args.verbose)
    else:
        check_links(Path(args.path), args.verbose)

if __name__ == "__main__":
    main()