# nano-vLLM OKF 지식 베이스 변경 이력 (Log)

All notable changes to the `.okf/00_nano_vllm` Knowledge Base will be documented in this file.
The format is based on ISO date ordering (newest first).

---

## [2026-08-14] - 학습 링크 정비 (module→composite→atomic 하향 트리)

### Added
- `index.md` 7-Module Map: 8개 Module 이름을 각 모듈 문서(`concepts/01_module/*.md`)로 연결,
  핵심 개념 목록을 실제 개념 파일로 연결 (`distributed_serving` → `distributed_serving_system.md` 포함).
- `meta/learning_path.md` 개념명을 실제 개념 파일로 연결 (옛 `00_nano_vllm/01_atomic_concepts` 경로 수리).

### Fixed
- **관계 블록 하향 링크 정비**: 모든 개념/모듈 파일의 관계 블록(PREREQUISITES / PREREQUISITE_OF /
  COMPOSED_OF / SYNERGY WITH 등)의 개념 id를 markdown 링크로 변환 (`okf_repair.py --linkify`,
  46개 파일) — module→composite→atomic 하향 트리 링크가 마크다운 뷰어·대시보드에서 모두 동작.
- **파서 정비**: 관계 블록이 비-항목 라인(헤딩/문단)에서 종료되도록 수정, 링크형 관계 항목
  (`[id](path)`) 파싱 지원 — 링크화 후에도 lint 0건 유지.

### Result
- `lint_knowledge_graph` 0건 유지, `okf_validate.py` PASS (0 errors)
- 대시보드 그래프 엣지 92 → 325 (관계 링크가 엣지로 반영)

---

## [2026-08-14] - 무결성 수리 (lint clean, Phase 8)

### Fixed
- **개념 순환 7건 제거**: atomic 개념들의 `COMPOSED_OF` 방향 오류를 `PREREQUISITE_OF`로 교정
  (`continuous_batching ↔ iteration_level_scheduling`, `memory_pool ↔ block_allocator ↔ swap_manager`,
  `distributed_kv ↔ fault_tolerance ↔ load_balancer`).
- **Dangling 참조 14건 재지정**: `distributed_serving` → `composite.distributed_serving`,
  `prefix_cache_manager` → `composite.prefix_cache_manager`,
  `composite.distributed_executor` → `composite.distributed_serving` (module_05/06/07, fault_tolerance,
  master_worker, shared_memory_ipc, tensor_parallelism).
- **관계 블록 산문 노이즈 제거**: `distributed_kv.md` 중간 이중 frontmatter 블록 제거 및 파일 전면 정리,
  `paged_kv_cache.md` `SYNERGY WITH` 블록 항목의 비-id 산문을 괄호 내부로 이동.
- **단방향 관계 22개 노드 역방향 미러링**: 누락된 `prerequisites` / `prerequisite_of` 선언을 frontmatter에 추가.

### Added
- 개념/규칙 노드 28개에 `sources` 증빙(evidence) 추가
  (vLLM PagedAttention 논문, arXiv:2209.06155).

### Result
- `lint_knowledge_graph` 0건 (순환·역방향·dangling·노이즈·증빙 부재 모두 해소)
- `okf_validate.py .okf/01_nano_vllm` PASS (0 errors)

---

## [2026-08-05] - OKF v0.2 구조 초기화

### Added
- `.okf/00_nano_vllm/index.md`: 메인 인덱스 및 7개 모듈 맵 정의
- `.okf/00_nano_vllm/log.md`: KB 변경 기록 일지 생성
