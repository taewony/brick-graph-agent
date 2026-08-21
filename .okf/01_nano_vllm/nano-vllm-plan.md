# BrickGraphAgent: nano‑vLLM 학습 가이드 에이전트 개발 계획 (수정안)

> *"From 14 labs to a living knowledge graph: OKF-based brick-by-brick learning with ActiveGraph-powered self-improving guide agent"*

---

## 📌 개요: 전환의 핵심

원래 개발 계획(`plan.md`)은 **보고서/제안서 논리 검증 및 지식 구축**을 목표로 했습니다. 이번 수정안에서는 **nano‑vLLM의 5~7개 모듈**을 OKF KB로 구축하고, 학습자가 **brick‑by‑brick**으로 개념을 마스터할 수 있도록 안내하는 **Learning Guide Agent**로 초점을 전환합니다.

**변경된 목표**:

| 항목 | 원래 계획 | 수정된 계획 (nano‑vLLM 학습 가이드) |
|---|---|---|
| **KB 대상** | 보고서/제안서 도메인 | nano‑vLLM 14개 모듈 (우선 5~7개 모듈) |
| **에이전트 역할** | 논리 모순 탐지 + 구조 개선 | **학습 경로 추적 + 개념 추천 + 이해도 평가** |
| **주요 출력** | 수정 제안 보고서 | **맞춤형 학습 경로 + 진척도 대시보드** |
| **성공 지표** | 논리 일관성 +12% | **학습 시간 단축 25% + 개념 정확도 향상** |

**이유**: nano‑vLLM의 구조적 복잡성은 OKF의 **계층적 개념 표현**과 **brick‑by‑brick** 접근법에 최적화되어 있습니다. 또한 ActiveGraph의 이벤트 소싱은 **학습자의 진척도 추적**과 **에이전트의 자가 개선**을 완벽하게 지원합니다.

---

## 🔄 nano‑vLLM Module 5~7 요약 (OKF 관점에서 재구성)

> **참고**: Module 3 (Continuous Batching)과 Module 4 (PagedAttention)는 **선행 조건(prerequisites)** 으로 포함됩니다. Module 5~7은 이 기반 위에 구축된 **고급 시스템 통합 및 최적화** 주제입니다.

### Module 5: Memory Management & Block Manager

| 개념 ID | 설명 | 선행 조건 |
|---|---|---|
| [`atomic.memory_pool`](concepts/03_atomic/memory_pool.md) | GPU 메모리 풀 추상화 | [`atomic.paged_kv_cache`](concepts/03_atomic/paged_kv_cache.md) |
| [`atomic.block_allocator`](concepts/03_atomic/block_allocator.md) | 블록 할당/해제 전략 (First‑Fit, Best‑Fit) | [`atomic.memory_pool`](concepts/03_atomic/memory_pool.md) |
| [`atomic.swap_manager`](concepts/03_atomic/swap_manager.md) | CPU‑GPU 스와핑 정책 | [`atomic.block_allocator`](concepts/03_atomic/block_allocator.md) |
| [`composite.block_manager`](concepts/02_composite/block_manager.md) | PagedAttention 블록 관리자 전체 | [`atomic.paged_kv_cache`](concepts/03_atomic/paged_kv_cache.md) + [`atomic.block_allocator`](concepts/03_atomic/block_allocator.md) |

### Module 6: Prefix Caching & Reuse

| 개념 ID | 설명 | 선행 조건 |
|---|---|---|
| [`atomic.prefix_cache`](concepts/03_atomic/prefix_cache.md) | 공통 프롬프트 접두사 KV Cache 재사용 | [`atomic.paged_kv_cache`](concepts/03_atomic/paged_kv_cache.md) |
| [`atomic.cache_hit_detection`](concepts/03_atomic/cache_hit_detection.md) | 캐시 히트 탐지 알고리즘 | [`atomic.prefix_cache`](concepts/03_atomic/prefix_cache.md) |
| `composite.caching_strategy` | 전체 캐싱 정책 (LRU, LFU 등) | [`atomic.prefix_cache`](concepts/03_atomic/prefix_cache.md) + [`atomic.swap_manager`](concepts/03_atomic/swap_manager.md) |

### Module 7: Distributed Serving & Load Balancing

| 개념 ID | 설명 | 선행 조건 |
|---|---|---|
| [`atomic.distributed_kv`](concepts/03_atomic/distributed_kv.md) | 분산 환경에서의 KV Cache 공유 | [`composite.block_manager`](concepts/02_composite/block_manager.md) |
| [`atomic.load_balancer`](concepts/03_atomic/load_balancer.md) | 요청 분산 및 노드 선택 전략 | [`atomic.continuous_batching`](concepts/03_atomic/continuous_batching.md) |
| [`atomic.fault_tolerance`](concepts/03_atomic/fault_tolerance.md) | 장애 복구 및 체크포인팅 | [`atomic.distributed_kv`](concepts/03_atomic/distributed_kv.md) |
| [`composite.distributed_serving`](concepts/02_composite/distributed_serving_system.md) | 전체 분산 서빙 시스템 | [`composite.serving_system`](concepts/02_composite/serving_system.md) + [`atomic.distributed_kv`](concepts/03_atomic/distributed_kv.md) |

---

## 🏗️ 수정된 디렉토리 구조

```
brick-graph-agent/
├── .okf/                                    # 📂 OKF v0.2 Knowledge Base
│   ├── index.md                             # KB 메인 인덱스 (nano‑vLLM 맵)
│   ├── log.md                               # 변경 이력
│   │
│   ├── 00_meta/
│   │   ├── learning_path.md                 # 전체 학습 경로 (Module 0→13)
│   │   └── glossary.md                      # 용어 사전
│   │
│   ├── 01_atomic_concepts/                  # 원자적 개념
│   │   ├── inference_only.md                # Module 0
│   │   ├── prefill_phase.md                 # Module 0
│   │   ├── decode_phase.md                  # Module 0
│   │   ├── kv_cache.md                      # Module 1
│   │   ├── attention.md                     # Module 1
│   │   ├── ffn.md                           # Module 1
│   │   ├── sampling.md                      # Module 1
│   │   ├── continuous_batching.md           # Module 3
│   │   ├── paged_kv_cache.md                # Module 4
│   │   ├── hybrid_scheduling.md             # Module 3
│   │   ├── memory_pool.md                   # Module 5 ⭐ NEW
│   │   ├── block_allocator.md               # Module 5 ⭐ NEW
│   │   ├── swap_manager.md                  # Module 5 ⭐ NEW
│   │   ├── prefix_cache.md                  # Module 6 ⭐ NEW
│   │   ├── cache_hit_detection.md           # Module 6 ⭐ NEW
│   │   ├── distributed_kv.md                # Module 7 ⭐ NEW
│   │   ├── load_balancer.md                 # Module 7 ⭐ NEW
│   │   └── fault_tolerance.md               # Module 7 ⭐ NEW
│   │
│   ├── 02_composite_concepts/               # 복합 개념
│   │   ├── inference_model.md               # Module 0/2
│   │   ├── autoregressive_loop.md           # Module 1
│   │   ├── decoder_layer.md                 # Module 1
│   │   ├── serving_system.md                # Module 3/4
│   │   ├── block_manager.md                 # Module 5 ⭐ NEW
│   │   ├── caching_strategy.md              # Module 6 ⭐ NEW
│   │   └── distributed_serving.md           # Module 7 ⭐ NEW
│   │
│   ├── 03_modules/                          # Module별 매핑
│   │   ├── module_00_foundational.md
│   │   ├── module_01_autoregressive.md
│   │   ├── module_02_kv_cache.md
│   │   ├── module_03_batching.md
│   │   ├── module_04_paged_attention.md
│   │   ├── module_05_memory_management.md   # ⭐ NEW
│   │   ├── module_06_prefix_caching.md      # ⭐ NEW
│   │   └── module_07_distributed_serving.md # ⭐ NEW
│   │
│   ├── 04_diagrams/                         # 다이어그램 참조
│   │   ├── prefill_decode_timeline.md
│   │   ├── decoder_layer_flow.md
│   │   └── memory_hierarchy.md              # ⭐ NEW
│   │
│   └── 05_learning_checkpoints/             # 학습 확인용 체크포인트
│       ├── checkpoint_03_batching.md
│       ├── checkpoint_04_paged_attention.md
│       ├── checkpoint_05_memory.md          # ⭐ NEW
│       ├── checkpoint_06_caching.md         # ⭐ NEW
│       └── checkpoint_07_distributed.md     # ⭐ NEW
│
├── src/                                     # 📂 파이썬 소스코드
│   ├── main.py                              # 에이전트 실행 진입점
│   │
│   ├── active_graph/                        # 🤖 ActiveGraph 레이어
│   │   ├── event_store.py
│   │   ├── topics.py
│   │   └── cache.py
│   │
│   ├── agent_core/                          # 🧱 코어 브릭
│   │   ├── injestor.py                      # Brick 01
│   │   ├── compiler.py                      # Brick 02
│   │   ├── questioner.py                    # Brick 03
│   │   └── browser.py                       # Brick 04
│   │
│   ├── agents/                              # 🧠 학습 가이드 에이전트
│   │   ├── base_agent.py
│   │   ├── learning_guide.py                # ⭐ 학습 경로 추천
│   │   ├── progress_tracker.py              # ⭐ 진척도 관리
│   │   ├── concept_explainer.py             # ⭐ 개념 설명 생성
│   │   └── mastery_evaluator.py             # ⭐ 이해도 평가
│   │
│   └── tools/                               # 🔧 유틸리티
│       ├── okf_validator.py
│       └── okf_visualizer.py
│
├── storage/                                 # 📂 런타임 데이터
│   ├── events.log
│   ├── learner_progress.json                # ⭐ 학습자 진척도
│   └── mastered_concepts.log                # ⭐ 마스터 기록
│
├── dist/                                    # 📂 웹 대시보드
│   └── index.html                           # 학습 진척도 시각화
│
├── experiments/                             # 📊 실험용
│   └── learner_simulation/                  # 학습자 시뮬레이션
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── plan_revised.md                          # 본 문서
```

---

## 📋 수정된 구현 로드맵 (Phase 1~5)

### Phase 1: 프로젝트 환경 및 OKF KB 초기 구축 (1주)

| 작업 | 설명 | 출력물 |
|---|---|---|
| **1.1 환경 설정** | Python 3.11+ 가상환경, `activegraph==1.10.0`, `pyyaml`, `marker-pdf`, `pandas`, `scipy`, `pytest` 설치 | `requirements.txt`, `pyproject.toml` |
| **1.2 OKF KB 초기화** | `.okf/` 디렉토리 생성, `index.md`, `log.md` 작성 | `.okf/index.md` |
| **1.3 nano‑vLLM Module 0~4 개념 OKF화** | Module 0~4의 10개 원자적 개념 + 4개 복합 개념 OKF 파일 작성 | `.okf/01_atomic_concepts/*.md`, `.okf/02_composite_concepts/*.md` |
| **1.4 개념 관계 정의** | `PREREQUISITE_OF`, `COMPOSED_OF` 관계 명시 | 각 OKF 파일의 `prerequisites`, `composes_into` 필드 |
| **1.5 ActiveGraph 이벤트 스토어 구현** | `event_store.py`, `topics.py`, `cache.py` 기본 구현 | `src/active_graph/` |

---

### Phase 2: Module 5~7 OKF화 및 Learning Path 완성 (1주)

> ⭐ **이번 수정안의 핵심 단계**

| 작업 | 설명 | 출력물 |
|---|---|---|
| **2.1 Module 5 OKF화** | Memory Management 개념 OKF 파일 작성 (`memory_pool.md`, `block_allocator.md`, `swap_manager.md`, `block_manager.md`) | `.okf/01_atomic_concepts/memory_pool.md` 등 |
| **2.2 Module 6 OKF화** | Prefix Caching 개념 OKF 파일 작성 (`prefix_cache.md`, `cache_hit_detection.md`, `caching_strategy.md`) | `.okf/01_atomic_concepts/prefix_cache.md` 등 |
| **2.3 Module 7 OKF화** | Distributed Serving 개념 OKF 파일 작성 (`distributed_kv.md`, `load_balancer.md`, `fault_tolerance.md`, `distributed_serving.md`) | `.okf/01_atomic_concepts/distributed_kv.md` 등 |
| **2.4 Learning Path 완성** | Module 3→5→6→7 전체 학습 경로 정의 | `.okf/00_meta/learning_path.md` |
| **2.5 Module 매핑 문서 작성** | 각 Module과 개념 간 매핑 테이블 | `.okf/03_modules/module_*.md` |
| **2.6 OKF 검증** | `okf_validator.py --strict` 실행 | 검증 통과 |

---

### Phase 3: Learning Guide Agent 구현 (2주)

| 작업 | 설명 | 출력물 |
|---|---|---|
| **3.1 Brick 01 (Injestor)** | `.okf/` 변경 감지 및 `KNOWLEDGE_MODIFIED` 이벤트 발행 | `src/agent_core/injestor.py` |
| **3.2 Brick 02 (Compiler)** | 개념 간 관계 그래프 컴파일 (`REFERENCES`, `PREREQUISITE_OF`, `COMPOSED_OF`) | `src/agent_core/compiler.py` |
| **3.3 Learning Guide Agent (`learning_guide.py`)** | - 학습자의 현재 진척도 조회<br>- 선행 조건이 충족된 다음 개념 추천<br>- `SUGGESTS_NEXT` 이벤트 발행 | `src/agents/learning_guide.py` |
| **3.4 Progress Tracker (`progress_tracker.py`)** | - 학습자의 개념 상태(`draft` → `learning` → `learned` → `mastered`) 관리<br>- `STATUS_UPDATED` 이벤트 발행 | `src/agents/progress_tracker.py` |
| **3.5 Concept Explainer (`concept_explainer.py`)** | - 특정 개념에 대한 맞춤형 설명 생성<br>- 관련 다이어그램 및 코드 예제 포함 | `src/agents/concept_explainer.py` |
| **3.6 Mastery Evaluator (`mastery_evaluator.py`)** | - 개념 이해도 평가 (퀴즈/실습 과제 자동 생성)<br>- 평가 결과에 따라 `mastered` 상태 부여 | `src/agents/mastery_evaluator.py` |

---

### Phase 4: 대시보드 및 시각화 (1주)

| 작업 | 설명 | 출력물 |
|---|---|---|
| **4.1 Brick 04 (Browser)** | ActiveGraph 이벤트를 웹소켓으로 스트리밍 | `src/agent_core/browser.py` |
| **4.2 학습 진척도 대시보드** | - 개념 그래프 시각화 (Cytoscape.js)<br>- 학습 상태별 색상 구분 (초록: mastered, 파랑: learned, 회색: draft)<br>- 추천 개념 하이라이트 | `dist/index.html` |
| **4.3 OKF Visualizer 연동** | `okf_visualize.py`로 정적 그래프 생성 | `dist/viz.html` |

---

### Phase 5: 실험 및 검증 (1주)

| 작업 | 설명 | 출력물 |
|---|---|---|
| **5.1 학습자 시뮬레이션** | 가상 학습자 에이전트가 Learning Guide Agent의 추천을 따라 학습하는 시뮬레이션 | `experiments/learner_simulation/` |
| **5.2 성능 측정** | - 학습 시간 (개념당 소요 시간)<br>- 개념 정확도 (마스터 평가 점수)<br>- 학습 경로 이탈률 | 실험 결과 CSV |
| **5.3 OKF 검증 및 CI 연동** | GitHub Actions에서 `.okf/` 자동 검증 | `.github/workflows/validate.yml` |

---

## 🧠 Learning Guide Agent 동작 상세

### 에이전트 흐름도

```
[학습자 상태 조회]
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Learning Guide Agent (src/agents/learning_guide.py)           │
│                                                                 │
│  1. 현재 학습자의 mastered/learned 개념 목록 조회              │
│  2. 전체 개념 그래프에서 선행 조건이 모두 충족된 개념 검색      │
│  3. 학습 경로(learning_path.md) 기준 우선순위 정렬             │
│  4. 상위 3개 개념 추천 (SUGGESTS_NEXT 이벤트 발행)            │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
[학습자가 개념 학습 시작] (status: draft → learning)
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Concept Explainer (src/agents/concept_explainer.py)           │
│                                                                 │
│  1. 해당 개념의 OKF 파일에서 설명, 다이어그램, 코드 예제 로드   │
│  2. 선행 개념과의 연결 관계 시각화                             │
│  3. 이해를 돕는 맞춤형 설명 생성 (LLM)                          │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
[학습자가 퀴즈/실습 완료]
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Mastery Evaluator (src/agents/mastery_evaluator.py)           │
│                                                                 │
│  1. 개념 이해도 평가 (퀴즈 정답률, 코드 실행 성공 여부)        │
│  2. 평가 점수 ≥ 80% → status: mastered                         │
│  3. 평가 점수 < 80% → status: learning (재학습 권장)           │
│  4. STATUS_UPDATED 이벤트 발행                                 │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
[진척도 업데이트] → Learning Guide Agent 재실행 (루프)
```

### OKF 상태 필드 활용

각 개념 파일의 `status` 필드를 학습 단계에 따라 업데이트:

```yaml
# .okf/01_atomic_concepts/continuous_batching.md
---
type: AtomicConcept
id: atomic.continuous_batching
title: Continuous Batching
status: mastered   # draft → learning → learned → mastered
learning_history:
  - started_at: 2026-08-05T10:00:00Z
  - mastered_at: 2026-08-06T14:30:00Z
  - attempts: 2
  - quiz_score: 0.85
---
```

---

## 📊 성공 측정 지표 (OKR)

| Objective | Key Result | 측정 방법 |
|---|---|---|
| **학습 효율성 향상** | 평균 개념 학습 시간 25% 단축 | 시뮬레이션 vs baseline (순차 학습) |
| **개념 이해도 향상** | 마스터 평가 평균 점수 80% 이상 | Mastery Evaluator 자동 채점 |
| **에이전트 추천 정확도** | 추천 개념 학습 성공률 75% 이상 | 추천→학습→마스터 완료 비율 |
| **OKF KB 품질** | OKF v0.2 검증 통과 (0 warning) | `okf_validator.py --strict` |
| **시스템 감사성** | 모든 에이전트 행동 이벤트 로그 기록 | `storage/events.log` 크기 > 0 |

---

## 🔧 기술 스택 요약

| 계층 | 기술 |
|---|---|
| **Runtime** | ActiveGraph 1.10.0 (event-sourced) |
| **Knowledge Base** | OKF v0.2 (Markdown + YAML frontmatter) |
| **LLM** | Claude Sonnet 4.5 (or equivalent) |
| **PDF Processing** | Marker (converter) |
| **Visualization** | Cytoscape.js, Mermaid |
| **Testing** | pytest, hypothesis |
| **CI/CD** | GitHub Actions |

---

## 📝 다음 단계 제안

1. **즉시 실행**: Phase 1.1~1.4 (환경 설정 및 Module 0~4 OKF화)
2. **2주 후**: Phase 2.1~2.3 (Module 5~7 OKF화) 완료 목표
3. **4주 후**: Phase 3.3~3.6 (Learning Guide Agent) MVP 완료
4. **6주 후**: Phase 5 (실험 및 검증) + 논문 초고 작성

---

> 💡 **핵심 인사이트**: 원래 설계의 **"논리적 모순 검출"** 기능은 Learning Guide Agent의 **"개념 간 관계 검증"** 으로 자연스럽게 확장됩니다. Critic Agent는 **"선행 조건 누락"** 또는 **"잘못된 의존성"** 을 탐지하여, 학습자가 비효율적인 경로로 학습하지 않도록 방지합니다. 이는 **nano‑vLLM의 복잡한 계층적 구조**를 학습하는 데 매우 적합합니다.