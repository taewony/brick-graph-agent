# BrickGraphAgent: 통합 개발 계획 (수정안)

> *“아키텍처는 설계되는 것이 아니라 도출되어야 한다. 그래프는 제어 표면(control surface)이지 조회 계층이 아니다.”*

> nano‑vLLM source: https://hackmd.io/9Ivogn3dRwm3WgA4Kl3fJQ

---

## 📌 1. 개요: 두 가지 검증 목표의 통합

BrickGraphAgent는 **OKF(Open Knowledge Format) v0.2 지식 베이스**와 **ActiveGraph 1.10.0 이벤트-소싱 런타임**을 결합한 자가 개선 에이전트 시스템입니다. 본 계획은 **두 가지 검증 목표**를 순차적으로 달성합니다.

| 단계 | 검증 목표 | 핵심 질문 | 우선순위 근거 |
|---|---|---|---|
| **Phase 1–4** | **nano‑vLLM 개념 관계 검증** (학습 가이드) | OKF 기반 계층적 지식 구조가 학습 효율을 개선하는가? | **Ground Truth 객관적 확보 용이**, 통계적 검정력 높음, Brick‑by‑Brick 철학에 가장 직접적 부합, 개발 속도 빠름 (6주) |
| **Phase 5** | **보고서/제안서 논리적 모순 검출** (도메인 확장) | 동일한 아키텍처가 비정형 문서의 논리적 오류도 검출/수정할 수 있는가? | nano‑vLLM 성공 후 **전이(transfer) 실험**으로 일반화 입증, 후속 논문 또는 논문 내 확장 섹션으로 활용 |

> 💡 **전략적 판단**: 주관적 레이블링에 의존하는 논리적 모순 검출을 먼저 진행하면 데이터셋 구축에 수개월이 소요되고 주석자 간 신뢰도 문제로 실험 실패 위험이 큽니다. 반면, nano‑vLLM의 개념 관계는 **기술적 사실(코드/수식으로 증명 가능)** 이므로 OKF로 기계 변환만으로 완성되며, **6주 내에 논문의 핵심 주장을 입증**할 수 있습니다.

---

## 🎯 2. 아키텍처 개요 및 핵심 원칙

```
┌───────────────────────────────────────────────────────────────────────────┐
│                       CONTROL PLANE (OKF KB Graph)                        │
│  - 온톨로지: AtomicConcept, CompositeConcept, Claim, Evidence, Premise    │
│  - 관계: PREREQUISITE_OF, COMPOSED_OF, SUPPORTS, CONTRADICTS, ENTAILS     │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                   ACTIVE GRAPH RUNTIME (ActiveGraph 1.10.0)                │
│  - Event Bus (storage/events.log, storage/events.log.index)               │
│  - Replay & Response Cache (Deterministic Execution)                      │
│  - Event-driven Triggers (@llm_behavior pattern matching)                │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    AGENT SUITE (Specialized Roles)                         │
│  - Builder Agent    : AtomicConcept → CompositeConcept assembly           │
│  - Learning Guide   : 학습 경로 추천 및 진척도 관리 (Phase 1–4 집중)      │
│  - Critic Agent     : 논리적 모순 및 관계 검증 (Phase 5 확장)             │
│  - Improver Agent   : 지식 트리 구조 최적화                               │
│  - Oracle Agent     : 실패 레짐 분류기                                    │
│  - Curator          : 인간 개발자 승인 및 거버넌스 게이트                 │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 3. 디렉토리 구조

```
brick-graph-agent/
├── .okf/                                    # 📂 OKF v0.2 Knowledge Base
│   ├── index.md                             # KB 메인 인덱스
│   ├── log.md                               # 변경 이력 (ISO 날짜, 최신순)
│   │
│   ├── 00_meta/
│   │   ├── learning_path.md                 # 전체 학습 경로
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
│   │   ├── memory_pool.md                   # Module 5
│   │   ├── block_allocator.md               # Module 5
│   │   ├── swap_manager.md                  # Module 5
│   │   ├── prefix_cache.md                  # Module 6
│   │   ├── cache_hit_detection.md           # Module 6
│   │   ├── distributed_kv.md                # Module 7
│   │   ├── load_balancer.md                 # Module 7
│   │   └── fault_tolerance.md               # Module 7
│   │
│   ├── 02_composite_concepts/               # 복합 개념
│   │   ├── inference_model.md
│   │   ├── autoregressive_loop.md
│   │   ├── decoder_layer.md
│   │   ├── serving_system.md
│   │   ├── block_manager.md                 # Module 5
│   │   ├── caching_strategy.md              # Module 6
│   │   └── distributed_serving.md           # Module 7
│   │
│   ├── 03_modules/                          # Module별 매핑
│   │   ├── module_00_foundational.md
│   │   ├── module_01_autoregressive.md
│   │   ├── module_02_kv_cache.md
│   │   ├── module_03_batching.md
│   │   ├── module_04_paged_attention.md
│   │   ├── module_05_memory_management.md
│   │   ├── module_06_prefix_caching.md
│   │   └── module_07_distributed_serving.md
│   │
│   ├── 04_diagrams/                         # 다이어그램 참조
│   │   ├── prefill_decode_timeline.md
│   │   └── memory_hierarchy.md
│   │
│   └── 05_domain_logic/                     # 논리 검증 규칙 (Phase 5)
│       ├── contradiction_patterns.md        # 모순 패턴 정의
│       └── inference_rules.md               # 추론 규칙 (ENTAILS, SUPPORTS)
│
├── src/
│   ├── main.py
│   ├── active_graph/                        # 🤖 ActiveGraph 레이어
│   │   ├── event_store.py
│   │   ├── topics.py
│   │   └── cache.py
│   ├── agent_core/                          # 🧱 코어 브릭
│   │   ├── injestor.py                      # Brick 01
│   │   ├── compiler.py                      # Brick 02
│   │   ├── questioner.py                    # Brick 03
│   │   └── browser.py                       # Brick 04
│   ├── agents/                              # 🧠 에이전트
│   │   ├── base_agent.py
│   │   ├── learning_guide.py                # ⭐ 학습 경로 추천 (Phase 1–4)
│   │   ├── progress_tracker.py              # ⭐ 진척도 관리
│   │   ├── concept_explainer.py             # ⭐ 개념 설명 생성
│   │   ├── mastery_evaluator.py             # ⭐ 이해도 평가
│   │   ├── critic.py                        # ⭐ 논리적 모순 탐지 (Phase 5)
│   │   ├── improver.py                      # ⭐ 구조 최적화 (Phase 5)
│   │   └── oracle.py                        # 실패 레짐 분류
│   └── tools/
│       ├── okf_validator.py
│       └── okf_visualizer.py
│
├── storage/
│   ├── events.log
│   ├── learner_progress.json                # ⭐ 학습자 진척도
│   └── mastered_concepts.log                # ⭐ 마스터 기록
│
├── dist/                                    # 웹 대시보드
│   └── index.html
│
├── experiments/
│   ├── nano_vllm/                           # Phase 1–4 실험
│   │   ├── splits/
│   │   ├── results/
│   │   └── analysis/
│   └── document_logic/                      # Phase 5 실험
│       ├── corpus/                          # 보고서/제안서 데이터셋
│       └── annotations/                     # 인간 주석
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 📋 4. 구현 로드맵

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

> ⭐ **nano‑vLLM 학습 가이드의 핵심 기반**

| 작업 | 설명 | 출력물 |
|---|---|---|
| **2.1 Module 5 OKF화** | Memory Management 개념 (`memory_pool.md`, `block_allocator.md`, `swap_manager.md`, `block_manager.md`) | `.okf/01_atomic_concepts/memory_pool.md` 등 |
| **2.2 Module 6 OKF화** | Prefix Caching 개념 (`prefix_cache.md`, `cache_hit_detection.md`, `caching_strategy.md`) | `.okf/01_atomic_concepts/prefix_cache.md` 등 |
| **2.3 Module 7 OKF화** | Distributed Serving 개념 (`distributed_kv.md`, `load_balancer.md`, `fault_tolerance.md`, `distributed_serving.md`) | `.okf/01_atomic_concepts/distributed_kv.md` 등 |
| **2.4 Learning Path 완성** | Module 0→7 전체 학습 경로 정의 | `.okf/00_meta/learning_path.md` |
| **2.5 OKF 검증** | `okf_validator.py --strict` 실행 | 검증 통과 |

---

### Phase 3: Learning Guide Agent 구현 (2주)

| 작업 | 설명 | 출력물 |
|---|---|---|
| **3.1 Brick 01 (Injestor)** | `.okf/` 변경 감지 및 `KNOWLEDGE_MODIFIED` 이벤트 발행 | `src/agent_core/injestor.py` |
| **3.2 Brick 02 (Compiler)** | 개념 간 관계 그래프 컴파일 (`REFERENCES`, `PREREQUISITE_OF`, `COMPOSED_OF`) | `src/agent_core/compiler.py` |
| **3.3 Learning Guide Agent** | - 학습자 진척도 조회<br>- 선행 조건 충족된 다음 개념 추천<br>- `SUGGESTS_NEXT` 이벤트 발행 | `src/agents/learning_guide.py` |
| **3.4 Progress Tracker** | 개념 상태(`draft`→`learning`→`learned`→`mastered`) 관리 | `src/agents/progress_tracker.py` |
| **3.5 Concept Explainer** | 특정 개념에 대한 맞춤형 설명 생성 (LLM) | `src/agents/concept_explainer.py` |
| **3.6 Mastery Evaluator** | 개념 이해도 평가 (퀴즈/실습 과제 자동 생성) | `src/agents/mastery_evaluator.py` |

---

### Phase 4: nano‑vLLM 실험 및 검증 (2주)

| 작업 | 설명 | 출력물 |
|---|---|---|
| **4.1 학습자 시뮬레이션** | 가상 학습자 에이전트가 Learning Guide Agent의 추천을 따라 학습 | `experiments/nano_vllm/simulation.py` |
| **4.2 성능 측정** | - 학습 시간 (개념당 소요 시간)<br>- 개념 정확도 (마스터 평가 점수)<br>- 학습 경로 이탈률 | 실험 결과 CSV |
| **4.3 통계 분석** | McNemar 검정, Bootstrap 신뢰구간 계산 | `experiments/nano_vllm/analysis/` |
| **4.4 대시보드 시각화** | 학습 진척도 그래프 (Cytoscape.js) | `dist/index.html` |

**예상 결과**: Baseline(순차 학습) 대비 학습 시간 **25% 단축**, 개념 마스터 정확도 **15% 향상**

---

### Phase 5: 보고서/제안서 논리적 모순 검출로 확장 (4주, 선택적)

> ⭐ **nano‑vLLM 성공 후 전이(transfer) 검증**

| 작업 | 설명 | 출력물 |
|---|---|---|
| **5.1 도메인 온톨로지 확장** | 보고서/제안서용 노드 타입 추가 (`Claim`, `Evidence`, `Premise`) 및 관계 (`SUPPORTS`, `CONTRADICTS`, `ENTAILS`, `BASED_ON`) | `.okf/05_domain_logic/` |
| **5.2 Critic Agent 구현** | - 직접 모순 탐지 (`CONTRADICTS`)<br>- 근거 없는 주장 탐지 (`BASED_ON` 누락)<br>- 순환 논증 탐지 | `src/agents/critic.py` |
| **5.3 Improver Agent 구현** | 트리 구조 최적화 (재구성, 분할, 병합, 재정렬) 제안 | `src/agents/improver.py` |
| **5.4 데이터셋 구축** | 보고서/제안서 300개 수집 및 인간 주석 (모순 레이블) | `experiments/document_logic/corpus/` |
| **5.5 Held-Out Gate 실험** | OPTIMIZE(50) / CONFIRM(100) / TEST(350) 분할로 개선안 검증 | `experiments/document_logic/results/` |
| **5.6 성능 측정** | F1-score, Precision, Recall (목표: +12% 개선) | 통계 보고서 |

---

## 🧠 5. Learning Guide Agent 동작 상세 (Phase 1–4 집중)

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
│  3. learning_path.md 기준 우선순위 정렬                        │
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
│  3. 맞춤형 설명 생성 (LLM)                                     │
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

## 🔬 6. 논문 실험 설계 (요약)

### 6.1 1차 실험: nano‑vLLM 학습 가이드 (Phase 1–4)

| 항목 | 내용 |
|---|---|
| **목표** | OKF 기반 계층적 지식 구조가 학습 효율을 개선하는지 검증 |
| **Baseline** | 학습자가 OKF 없이 nano‑vLLM 문서를 순차적으로 학습 (또는 RAG) |
| **Ours** | Learning Guide Agent가 OKF 그래프 기반 최적 학습 경로 추천 + Mastery Evaluator로 이해도 검증 |
| **측정 지표** | 전체 Module 0~7 학습 완료 시간, 최종 종합 평가 점수 |
| **통계** | McNemar 검정, Bootstrap 신뢰구간 (95%) |
| **예상 결과** | 학습 시간 25% 단축, 정확도 15% 향상 |

### 6.2 2차 실험: 보고서/제안서 논리적 모순 검출 (Phase 5)

| 항목 | 내용 |
|---|---|
| **목표** | 동일 아키텍처가 비정형 문서의 논리적 오류도 검출/수정 가능한지 검증 |
| **Baseline** | GPT-4 few-shot (비-그래프 기반) |
| **Ours** | Critic Agent + Improver Agent (OKF 그래프 기반) |
| **측정 지표** | F1-score, Precision, Recall |
| **통계** | Paired t-test |
| **예상 결과** | F1-score +12% 개선 |

---

## 📊 7. 성공 측정 지표 (OKR)

| Objective | Key Result | 측정 방법 | 목표 |
|---|---|---|---|
| **학습 효율성 향상** | 평균 개념 학습 시간 단축 | 시뮬레이션 vs baseline | **25%** |
| **개념 이해도 향상** | 마스터 평가 평균 점수 | Mastery Evaluator 자동 채점 | **80% 이상** |
| **에이전트 추천 정확도** | 추천 개념 학습 성공률 | 추천→학습→마스터 완료 비율 | **75% 이상** |
| **논리적 모순 검출 정확도** | F1-score | 인간 주석 vs 에이전트 탐지 | **+12%** (baseline 대비) |
| **OKF KB 품질** | OKF v0.2 검증 통과 | `okf_validator.py --strict` | **0 warning** |
| **시스템 감사성** | 모든 에이전트 행동 이벤트 로그 | `storage/events.log` 크기 | **> 0** |

---

## 🔧 8. 기술 스택 요약

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

## 📝 9. 논문 작성 전략

### 9.1 Phase 1–4 완료 후 (6주): 1차 논문 초고

- **주요 기여(Contribution)**:
  1. OKF + ActiveGraph 기반 **계층적 지식 그래프**를 학습 제어 평면(control plane)으로 활용
  2. **Brick‑by‑Brick 학습 경로 최적화** 에이전트 (Learning Guide Agent)
  3. **Mastery Evaluator**를 통한 객관적 이해도 검증
  4. nano‑vLLM 도메인에서의 **학습 효율 25% 개선** 실증

- **논문 섹션**:
  - *Introduction*: AI 시대의 이해(understanding) 병목 문제 제기
  - *System Design*: OKF KB + ActiveGraph 아키텍처 상세
  - *Experiments*: nano‑vLLM 학습 가이드 실험 결과 (Table 1, 2; Figure 3)
  - *Discussion*: Brick‑by‑Brick 접근법의 일반화 가능성

### 9.2 Phase 5 완료 후 (10주): 2차 논문 (확장/후속)

- **추가 기여**:
  5. **논리적 모순 검출 및 수정 제안**으로의 도메인 전이(transfer) 검증
  6. 보고서/제안서 도메인에서의 **F1-score +12% 개선** 실증

- **논문 활용**:
  - 1차 논문의 *Generalization* 섹션에 추가
  - 또는 별도 후속 논문으로 분리

---

> 💡 **최종 권고**: nano‑vLLM 검증을 **6주 내에 완료**하여 논문의 핵심 주장을 확보하십시오. 여기서 확보된 실험 데이터와 통계적 유의성은 논문의 뼈대가 됩니다. 이후 논리적 모순 검출은 **"동일 아키텍처의 다른 도메인 적용"** 사례로 자연스럽게 확장하면 됩니다. 이는 학계에서 가장 선호하는 **"깔끔한 실험 설계 → 명확한 결과 → 강력한 주장"** 의 구조를 완성하는 지름길입니다.