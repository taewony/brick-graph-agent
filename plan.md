# BrickGraphAgent: 통합 개발 계획 (ActiveGraph 1.10.0 기반)

> *"아키텍처는 설계되는 것이 아니라 도출되어야 한다. 그래프는 제어 표면(control surface)이지 조회 계층이 아니다."*
> 
> **nano‑vLLM Reference Source**: [https://hackmd.io/9Ivogn3dRwm3WgA4Kl3fJQ](https://hackmd.io/9Ivogn3dRwm3WgA4Kl3fJQ)

---

## 📌 1. 개요: 두 가지 검증 목표의 통합

**BrickGraphAgent**는 OKF(Open Knowledge Format) v0.2 지식 베이스와 ActiveGraph 1.10.0 이벤트-소싱 런타임을 결합한 자가 개선 에이전트 시스템입니다. 본 계획은 **nano‑vLLM 7개 모듈을 `.okf` 기반 brick-by-brick 접근법으로 구축**하고, 이후 보고서/제안서의 논리적 모순을 검출하고 개선하며, 실적보고서를 생성하는 시스템으로 확장하는 **단계별 개발 로드맵**을 제공합니다.

| 단계 | 검증 목표 | 핵심 질문 | 우선순위 근거 |
|---|---|---|---|
| **Phase 1–4** | **nano‑vLLM 7개 모듈 OKF 구축 & 학습 가이드** | OKF 기반 계층적 지식 구조가 학습 효율을 개선하는가? | **Ground Truth 객관적 확보 용이**, 통계적 검정력 높음, Brick‑by‑Brick 철학에 직접 부합 (6주 소요) |
| **Phase 5** | **보고서/제안서 논리적 모순 검출** (도메인 1차 확장) | 동일한 아키텍처가 비정형 문서의 논리적 오류도 검출/수정할 수 있는가? | nano‑vLLM 성공 후 **전이(transfer) 실험**으로 일반화 입증 |
| **Phase 6** | **보고서/제안서 기준 실적보고서 생성** (도메인 2차 확장) | 동일한 아키텍처가 기준 문서 대비 fact 및 실적 data 기반으로 실적보고서를 작성할 수 있는가? | 1차 확장 성공 후 **전이(transfer) 실험**으로 일반화 입증 |
---

## 🎯 2. 아키텍처 개요 및 핵심 원칙

1. **아키텍처는 궤적(traces)에서 도출된다**: 모든 에이전트 행동 및 시스템 진화는 불변 이벤트 로그에 기록됩니다.
2. **결정론적 영역과 에이전틱 영역의 분리**: 수식/관계 검증은 결정론적 파이프라인으로, 추론과 제안은 에이전트가 담당합니다.
3. **그래프는 제어 표면(Control Surface)이다**: OKF KB 지식 그래프가 에이전트의 거버넌스와 권한을 통제합니다.
4. **보류 평가 게이트 (Held-Out Gate)**: 후보 수정안은 Validation set (OPTIMIZE -> CONFIRM -> TEST)에서 승인된 경우에만 프로모션됩니다.

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
│  - Learning Guide   : nano-vLLM 7개 모듈 학습 경로 추천 & 진척도 관리      │
│  - Critic Agent     : 논리적 모순 및 관계 검증                            │
│  - Improver Agent   : 지식 트리 구조 최적화 (Recomposition/Splitting)     │
│  - Oracle Agent     : 실패 레짐 분류기 (Failure Regime Classifier)        │
│  - Curator          : 인간 개발자 승인 및 거버넌스 게이트                 │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 3. `.okf` 디렉토리 및 프로젝트 구조 (7개 모듈 매핑)

```
brick-graph-agent/
└─ .okf/                                     # 📂 OKF v0.2 Knowledge Base (Root)
  └ 00_nano_vllm/                            # 📂 nano-vllm Knowledge Base (Root)
    ├── index.md                             # KB 메인 인덱스
    ├── log.md                               # 변경 이력 (ISO 날짜, 최신순)
    │
    ├── 00_meta/                             # 📂 메타 정보
    │   ├── learning_path.md                 # 전체 학습 경로 (Module 00 → 07)
    │   └── glossary.md                      # 용어 사전
    │
    ├── 01_modules/                          # 📂 nano-vLLM 7개 모듈 명세
    │   ├── module_00_foundational.md        # Module 00: Foundational Concepts
    │   ├── module_01_autoregressive.md      # Module 01: Autoregressive Generation
    │   ├── module_02_basic_kv_cache.md      # Module 02: Basic KV Cache Engine
    │   ├── module_03_continuous_batching.md # Module 03: Continuous Batching & Scheduling
    │   ├── module_04_paged_attention.md     # Module 04: PagedAttention
    │   ├── module_05_memory_management.md   # Module 05: Memory Management & Allocation
    │   ├── module_06_prefix_caching.md      # Module 06: Advanced Prefix Caching
    │   └── module_07_distributed_serving.md # Module 07: Multi-GPU & Distributed Serving
    │
    ├── 02_composite_concepts/               # 📂 복합 개념 (Brick-by-Brick 조립)
    │   ├── inference_model.md               # Module 00 (기초 추론 파이프라인)
    │   ├── autoregressive_loop.md           # Module 01 (자기회귀 생성 루프)
    │   ├── decoder_layer.md                 # Module 01 (트랜스포머 디코더 계층)
    │   ├── static_cache_manager.md          # Module 02 (정적 KV 캐시 관리자)
    │   ├── dynamic_batcher.md               # Module 03 (동적 배치 엔진)
    │   ├── serving_system.md                # Module 03-04 (Continuous Batching + PagedAttention 통합)
    │   ├── block_manager.md                 # Module 05 (물리적 블록 관리자)
    │   ├── caching_strategy.md              # Module 06 (프리픽스 캐싱 전략 관리자)
    │   └── distributed_serving_system.md    # Module 07 (분산 서빙 시스템)
    │
    └── 03_atomic_concepts/                  # 📂 원자적 개념 (계층적 모듈별 분류)
        │
        ├── 00_foundational                 # 🧱 Module 00: 기초 개념
        │   ├── inference_only.md
        │   ├── prefill_phase.md
        │   └── decode_phase.md
        │
        ├── 01_autoregressive               # 🧱 Module 01: 자기회귀 생성
        │   ├── kv_cache.md
        │   ├── attention.md
        │   ├── ffn.md
        │   └── sampling.md
        │
        ├── 02_static_cache                 # 🧱 Module 02: 정적 KV 캐시 (1세대)
        │   ├── static_kv_cache.md
        │   └── seq_len_budget.md
        │
        ├── 03_batching                     # 🧱 Module 03: 연속 배치 및 스케줄링
        │   ├── continuous_batching.md
        │   ├── iteration_level_scheduling.md
        │   ├── max_batch_size.md
        │   └── length_aware_scheduler.md
        │
        ├── 04_paged_attention              # 🧱 Module 04: 페이지드 어텐션 (논리-물리 분리)
        │   ├── paged_kv_cache.md            # ✅ 신규 추가됨
        │   ├── block_table.md               # ✅ 신규 추가됨
        │   └── slot_mapping.md              # ✅ 신규 추가됨
        │
        ├── 05_memory_management            # 🧱 Module 05: 메모리 풀 및 블록 관리
        │   ├── memory_pool.md
        │   ├── block_allocator.md
        │   └── swap_manager.md
        │
        ├── 06_prefix_caching               # 🧱 Module 06: 프리픽스 캐싱
        │   ├── prefix_cache.md
        │   └── cache_hit_detection.md
        │
        └── 07_distributed_serving          # 🧱 Module 07: 분산 서빙 (Tensor Parallelism)
            ├── tensor_parallelism.md
            ├── nccl_communicator.md
            ├── master_worker.md
            ├── shared_memory_ipc.md
            ├── distributed_kv.md
            ├── fault_tolerance.md
            └── load_balancer.md
```
```
brick-graph-agent/
└── .okf/  
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
│   ├── agents/                              # 🧠 전문 에이전트
│   │   ├── base_agent.py
│   │   ├── builder.py                       # CompositeConcept 조립
│   │   ├── learning_guide.py                # ⭐ nano-vLLM 학습 경로 추천 (Phase 1–4)
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
│   ├── learner_progress.json                # 학습자 진척도
│   └── mastered_concepts.log                # 마스터 기록
│
├── dist/                                    # 웹 대시보드
│   └── index.html
│
├── experiments/
│   ├── nano_vllm/                           # Phase 1–4 nano-vLLM 실험
│   │   ├── splits/
│   │   ├── results/
│   │   └── analysis/
│   └── document_logic/                      # Phase 5 보고서/제안서 실험
│       ├── corpus/
│       └── annotations/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 📋 4. 구현 로드맵

### Phase 1: 프로젝트 환경 및 nano-vLLM Module 0~4 OKF 구축 (1주)
- [v] **1.1 환경 설정**
  - Python 3.11+ 가상환경, `activegraph==1.10.0`, `pyyaml`, `marker-pdf`, `pandas`, `scipy`, `pytest` 설치
- [ ] **1.2 OKF KB 구조 초기화**
  - `.okf/` 디렉토리 생성 및 `index.md`, `log.md`, `00_meta/` 작성
- [v] **1.3 nano‑vLLM Module 0~4 개념 OKF 구축**
  - Module 0~4 원자적 개념 (`inference_only`, `prefill_phase`, `decode_phase`, `kv_cache`, `attention`, `ffn`, `sampling`, `continuous_batching`, `paged_kv_cache`) 파일 작성 및 `PREREQUISITE_OF` 관계 정의
- [ ] **1.4 ActiveGraph 이벤트 스토어 구현 (`src/active_graph/`)**
  - `event_store.py`, `topics.py`, `cache.py` 구현

### Phase 2: nano-vLLM Module 5~7 OKF 구축 및 Learning Path 완성 (1주)
- [v] **2.1 Module 5 (Memory Management) OKF 구축**
  - `memory_pool.md`, `block_allocator.md`, `swap_manager.md`, `block_manager.md`
- [v] **2.2 Module 6 (Prefix Caching) OKF 구축**
  - `prefix_cache.md`, `cache_hit_detection.md`, `caching_strategy.md`
- [v] **2.3 Module 7 (Distributed Serving) OKF 구축**
  - `distributed_kv.md`, `load_balancer.md`, `fault_tolerance.md`, `distributed_serving.md`
- [ ] **2.4 Learning Path 완성 & OKF 검증**
  - `.okf/00_meta/learning_path.md` 완성 및 `okf_validator.py --strict` 검증 통과

### Phase 3: Agent Core & Learning Guide Agent 구현 (2주)
- [ ] **3.1 ActiveGraph Core Bricks (`src/agent_core/`)**
  - Brick 01 (Injestor), Brick 02 (Compiler), Brick 03 (Questioner), Brick 04 (Browser)
- [ ] **3.2 Learning Guide & Progress Tracker (`src/agents/`)**
  - `learning_guide.py`: 선행조건 기반 개념 추천 (`SUGGESTS_NEXT` 이벤트 발행)
  - `progress_tracker.py`: 개념 상태 (`draft` -> `learning` -> `learned` -> `mastered`) 추적
  - `concept_explainer.py` & `mastery_evaluator.py`: 맞춤형 설명 및 이해도 퀴즈/평가

### Phase 4: nano‑vLLM 시뮬레이션 및 검증 (2주)

- [ ] **4.1 학습자 에이전트 시뮬레이션**
  - Baseline (순차 학습 / RAG) vs Ours (Learning Guide Agent 추천) 학습 효율 비교
- [ ] **4.2 성능 측정 및 통계 분석**
  - 학습 소요 시간 (25% 단축 목표), 마스터 정확도 (15% 향상 목표), McNemar 검정 및 Bootstrap 신뢰구간 분석
- [ ] **4.3 대시보드 시각화**
  - `dist/index.html` 지식 그래프 실시간 대시보드 연동

#### 🔧 4.4 도구 기반 입체적 학습 환경 구축 (Tool‑Driven 3D Learning Environment)

OKF KB로 구축된 nano‑vLLM 7개 모듈의 **brick‑by‑brick 계층적 지식 구조**를 효과적으로 학습하고, 문서 간 상호 연결성(Link Connectivity)을 체계적으로 검증하기 위해 다음 두 가지 도구를 활용합니다.

##### 4.4.1 `okf_link_check.py` — 계층적 링크 연결성 검사기 (Link Connectivity Checker)

**위치**: `src/tools/okf_link_check.py`

| 기능 | 설명 |
| :--- | :--- |
| **전체 링크 검사** | `.okf/` 내 모든 `.md` 파일의 위키 링크(`[[...]]`), 마크다운 링크(`[...](...)`), `file:///` 절대 경로, `Relationships` 섹션의 개념 ID를 종합 검사 |
| **모듈별 검사** | `-m 1` ~ `-m 7` 옵션으로 특정 Module의 관련 파일만 검사하고, **재귀적(Recursive) 확장**을 통해 해당 Module이 참조하는 Composite 및 Atomic 파일까지 자동 추적 |
| **자기 자신 링크 제거** | 자기 자신을 가리키는 링크는 출력에서 완전히 필터링하여 불필요한 노이즈 제거 |
| **중복 링크 제거** | 동일한 `(소스파일, 대상파일)` 쌍은 한 번만 출력 |
| **간결한 경로 출력** | `file:///D:/code/...` 접두사를 제거하고 `root` 기준 **상대 경로**로 깔끔하게 표시 |
| **상세 모드 (`-v`)** | 유효한 링크까지 모두 출력하여 전체 연결 관계를 한눈에 파악 |

**사용 예시**:
```bash
# Module 1 (Autoregressive Generation) 검사 + 유효한 링크 출력
python src/tools/okf_link_check.py -b 0 -m 1 -v

# Module 4 (PagedAttention) 검사
python src/tools/okf_link_check.py -b 0 -m 4

# 전체 검사 (깨진 링크만 출력)
python src/tools/okf_link_check.py -b 0
```

### Phase 5: 보고서/제안서 논리적 모순 검출 확장 (4주, 선택적)
- [ ] **5.1 도메인 온톨로지 확장**
  - `Claim`, `Evidence`, `Premise` 노드 및 `SUPPORTS`, `CONTRADICTS`, `ENTAILS`, `BASED_ON` 관계 추가
- [ ] **5.2 Critic & Improver Agent 구현**
  - Direct contradiction, unsupported claim, circular reasoning 탐지 및 트리 최적화 제안
- [ ] **5.3 Held-Out Gate 파이프라인 & 전이 검증**
  - 500개 문서 커퍼스 split (OPTIMIZE 50 / CONFIRM 100 / TEST 350) 기반 F1-score +12% 개선 검증

### Phase 6: 기준 보고서/제안서 대비 실적보고서 작성 (TBD)

---

## 🧪 검증 및 평가 전략 (Verification & OKRs)

1. **OKF 검증**:
   - `python -m src.tools.okf_validator .okf --strict` (Warning 0건 통과)
2. **단위 및 통합 테스트**:
   - `pytest tests/test_bricks.py` 및 `pytest tests/test_agents.py`
3. **핵심 성과 지표 (OKRs)**:
   - **학습 소요 시간**: Baseline 대비 **25% 단축**
   - **개념 마스터 평가 점수**: **80% 이상** 달성
   - **논리적 모순 검출 F1-Score**: Baseline 대비 **+12%** 향상 (Phase 5)

