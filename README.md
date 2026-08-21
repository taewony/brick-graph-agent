# brick.agent — OKF 지식 Q&A + 자연어 DB Q&A + 자가 개선 에이전트

> **brick.agent**는 두 가지 자연어 서비스를 하나의 이벤트-소싱 런타임 위에서 제공합니다.
> 1. **OKF Q&A** — 사내 문서를 OKF(Open Knowledge Format) wiki 지식베이스로 구축하고 질문에 답합니다.
> 2. **SQL Q&A** — 자연어 질문을 SQL로 변환해 SQLite 데이터베이스에서 실제 조회합니다.
>
> 그 위에 **멘토(자가 개선) 루프**와 **LLM observability**(이벤트 로그 기반 비용·지연·재생)가 얹혀 있습니다.

---

## 🚦 구현 상태

| 항목 | 상태 |
|---|---|
| OKF 타겟 (ingest / lint / ask) | ✅ Phase 2–3 완료 |
| SQL 타겟 (스키마 인코딩 → 컬럼 스코어링 → 프롬프트 → 드래프트) | ✅ Phase 1 완료 |
| 런타임 공유 서비스 (reader_registry, embedder, guardrails, loader, history_logger) | ✅ Phase 4 완료 |
| Observability & 이벤트 소싱 (SQLite EventStore, llm.* 이벤트, 결정적 재생) | ✅ Phase 5 완료 |
| Request Router + brick-agent CLI | ✅ Phase 6 완료 |
| Mentor 루프 (OKF/SQL 자가 개선 + LLM 사용량 관측) | ✅ Phase 7 완료 |
| 통합·테스트·KB 정리 (lint clean, README/패키징) | ✅ Phase 8 완료 |
| 테스트 | ✅ 69 tests 통과 (`python -m pytest tests/ -q`) |

---

## 🏗️ 아키텍처

```
┌───────────────────────────────────────────────────────────────┐
│  brick-agent CLI  (src/agents/brick_agent.py)                │
│    ask sql / ask okf / ask(router) / browse / ingest / lint  │
│    / db seed / mentor status    --real --store --trace       │
├───────────────────────────────────────────────────────────────┤
│  Router (src/agents/router.py) — 역할 자동 분류              │
│    sql | okf_ingest | okf_lint | okf_ask | mentor            │
├───────────────────────────────────────────────────────────────┤
│  ActiveGraph Runtime (event-driven behaviors)                │
│   sql_agent.*     schema.encoded → columns.scored →          │
│                   prompt.assembled → query.drafted           │
│   okf_agent.*     ingest/lint 체인 + ask 체인                │
│                   (context.assembled → llm.* → answer)       │
│   loop.*          mentor 자가 개선 루프                      │
├───────────────────────────────────────────────────────────────┤
│  Shared runtime services (src/runtime/)                      │
│   reader_registry · embedder · guardrails · history_logger   │
│   loader · event_store · observability · replay              │
├───────────────────────────────────────────────────────────────┤
│  Event Store (data/events.db, SQLite, append-only)           │
│   모든 run의 이벤트 로그 + llm.requested/llm.responded        │
│   (model · prompt_hash · answer · latency) — 결정적 재생 지원 │
└───────────────────────────────────────────────────────────────┘
```

모든 실행은 append-only 이벤트 로그로 기록되며, LLM 호출은 `llm.requested` → `llm.responded` 이벤트로
저장됩니다(프롬프트 해시 + 답변 포함). 저장된 답변은 `ReplayReader`로 **LLM 재호출 없이** 재생됩니다.

---

## 📂 실제 디렉토리 구조

```
brick-graph-agent/
├── src/
│   ├── core/
│   │   ├── agent/               # 공통 코어 (embedder, transforms, ...)
│   │   ├── eval/                # 평가 추상화 (Outcome, EvalResult, Reader)
│   │   ├── loop/                # 멘토 자가 개선 루프 (runner, behaviors, gates, regimes)
│   │   └── targets/
│   │       ├── sql/             # SQL 타겟 (agent/, action_space, eval, taxonomy, ...)
│   │       └── okf/             # OKF 타겟 (outcome, taxonomy, action_space,
│   │                             #   prompt_transforms, eval, target) — Phase 2
│   ├── agents/
│   │   ├── router.py            # 역할 자동 분류
│   │   ├── brick_agent.py       # CLI 진입점
│   │   └── okf/                 # OKF 런타임 behaviors (ingest/lint/ask) + agent
│   ├── runtime/                 # 공유 서비스 (Phase 4–5)
│   │   ├── reader_registry.py   #   Reader 등록/호출 래퍼
│   │   ├── embedder.py          #   임베딩 서비스
│   │   ├── guardrails.py        #   config/agent_model/guardrails.yaml 적용
│   │   ├── history_logger.py    #   .okf/*/history.yaml canonical 연산자 기록
│   │   ├── loader.py            #   OKF 번들 → 그래프 로드/materialize
│   │   ├── event_store.py       #   SQLiteEventStore + run 메타데이터
│   │   ├── observability.py     #   llm.* 이벤트 seam + 사용량 집계
│   │   └── replay.py            #   결정적 재생 + ReplayReader
│   ├── okf/                     # OKF 컴파일러 (validator, ir, history, compiler)
│   └── tools/
│       ├── seed_store_front.py  # data/store_front.db 시드 (.okf/02_store_front 기반)
│       ├── okf_link_check.py    # 링크 검사기
│       └── okf_visualizer.py    # 시각화 도구
├── config/agent_model/guardrails.yaml   # SQL/OKF/Mentor 가드레일
├── data/
│   ├── store_front.db           # SQL Q&A 데모 DB (5개 테이블, FK 포함)
│   └── events.db                # 이벤트 소싱 로그 (모든 run)
├── .okf/
│   ├── 00_agent_model/          # 통합 에이전트 스펙
│   │   ├── events.yaml          #   canonical 이벤트 어휘
│   │   ├── behaviors.spec.yaml  #   통합 행동 스펙 (handler 바인딩)
│   │   ├── concepts/observability.md  #   관측 가능성 명세
│   │   └── history.yaml         #   에이전트 모델 진화 기록
│   ├── 01_nano_vllm/            # 데모 지식베이스 (38개 개념)
│   └── 02_store_front/          # SQL 데모 도메인 (tools/skills/decisions)
├── tests/                       # 65 tests
├── user-manual.md               # CLI 상세 사용법 (본 문서의 사용법 확장판)
└── brick-agent-plan.md          # 개발 계획 (Phase 1–8)
```

---

## 🚀 빠른 시작 (API 키 불필요)

```powershell
cd D:\code\brick-graph-agent

# 1) SQL 데모 DB 준비 (.okf/02_store_front 시드)
.\.wenv\Scripts\python.exe -m src.agents.brick_agent db seed

# 2) SQL Q&A — 데이터베이스 조회
.\.wenv\Scripts\python.exe -m src.agents.brick_agent ask sql "How many stable tools are in the store front?"
#   SQL   : SELECT COUNT(*) FROM tools WHERE status = 'stable'
#   row   : (5,)

# 3) OKF Q&A — 지식베이스 질문
.\.wenv\Scripts\python.exe -m src.agents.brick_agent ask okf "How does prefill_phase relate to decode_phase?"
#   answer: prefill_phase is a prerequisite of decode_phase

# 4) 역할 자동 분류 (router)
.\.wenv\Scripts\python.exe -m src.agents.brick_agent ask "List the stable decisions"

# 5) LLM 관측 트레일 확인
.\.wenv\Scripts\python.exe -m src.agents.brick_agent ask okf "What is the kv_cache?" --trace

# 6) KB 웹 대시보드 브라우징 (docs/01_nano_vllm/index.html 생성 + 열기)
.\.wenv\Scripts\python.exe -m src.agents.brick_agent browse --kb 01_nano_vllm
```

**실제 LLM 사용** (`--real`): `ANTHROPIC_API_KEY`를 설정하면 자유로운 질문에 답할 수 있습니다.
자세한 CLI 참조·데모 질문 목록·문제 해결은 **[user-manual.md](user-manual.md)** 참조.

---

## 🔍 관측 가능성 & 결정적 재생

- 모든 run의 이벤트 로그가 `data/events.db`에 append-only로 영속됩니다.
- `--trace`: `llm.requested`(model, prompt_hash) / `llm.responded`(answer, latency, error) 출력.
- `mentor status`: 파이프라인·가드레일·저장 run·**LLM 사용량**(호출 수, 평균 지연, 에러) 집계.
- `ReplayReader`(prompt_hash 키)로 검증/재실행 시 **LLM 재호출 없이** 기록된 답변 서빙 (held-out 검증 결정적).

```powershell
.\.wenv\Scripts\python.exe -m src.agents.brick_agent mentor status
# sql pipeline : [] | okf pipeline : [inject_concept_tree, inject_rules, trim_context]
# llm usage    : calls=10 avg_latency=0.0s errors=0
```

---

## 🧠 Mentor 루프 (자가 개선)

`run_loop(target=...)`이 OKF/SQL 타겟 모두를 구동합니다:
`baseline → diagnose(regime histogram) → hypothesize(draft) → static/sandbox/eval-diff 게이트 → promote/discard → attribute → iterate/stop`.

- OKF 택소노미: `concept-cycle`(벽) · `rule-schema-mismatch` · `concept-orphan` · `ambiguous-trigger` · `unclassified`
- 신규 실패 클러스터는 `OkfTaxonomy.register_regime`으로 확장 (병렬 택소노미 금지)
- 검증은 `ReplayReader`로 결정적 재생 (Phase 5)

---

## 📋 OKF KB 감사 현황 (lint)

`brick-agent lint --kb-id <id> --kb-path .okf/01_nano_vllm` 또는 `lint_knowledge_graph()`로
8개 결정적 디텍터가 KB 무결성을 검사합니다.

| 코드 | 의미 | 01_nano_vllm (수리 전 → 후) |
|---|---|---|
| `cyclic_concept` | 개념 순환 의존 | 7 → 0 |
| `inverse_relationship` | 관계 단방향 기재 (논리적 모순) | 53 → 0 |
| `dangling_reference` | 존재하지 않는 노드 참조 (body 블록) | 14 → 0 |
| `relationship_noise` | 관계 블록의 산문/쓰레기 파싱 | 4 → 0 |
| `missing_evidence` | `sources` 증빙 문서 부재 | 28 → 0 |
| `concept_orphan` / `rule_schema_reference` / `ambiguous_trigger` | 구조·규칙·트리거 신호 | — |
| **합계** | | **106 → 0 (lint clean)** |

> **Phase 8 완료**: `.okf/01_nano_vllm` 무결성 수리(`src/tools/okf_repair.py`, 멱등)로
> 순환·dangling·노이즈·단방향 관계·증빙 부재를 모두 해소 — `brick-agent lint` 결과
> `valid=True n_errors=0`, `okf_validate.py` 0 errors. 복구 툴: `make repair`.

---

## 🗺️ 로드맵

- [x] Phase 1 — 기존 core 검증 (SQL)
- [x] Phase 2 — OKF 타겟 설계 (`src/core/targets/okf/`)
- [x] Phase 3 — OKF 런타임 behaviors (`src/agents/okf/`)
- [x] Phase 4 — 런타임 공유 서비스 (`src/runtime/`)
- [x] Phase 5 — Observability & 이벤트 소싱 (EventStore, llm.*, replay)
- [x] Phase 6 — Router + brick-agent CLI
- [x] Phase 7 — Mentor 통합 (OKF/SQL 자가 개선 + LLM 사용량 관측)
- [x] Phase 8 — 통합 테스트, `.okf/01_nano_vllm` lint clean (순환·dangling·증빙 106건 수리),
  Makefile/패키징 — **ALL PHASES COMPLETE**

---

## 📖 관련 자료

- [user-manual.md](user-manual.md) — CLI 상세 사용법 (SQL/OKF Q&A, --real/--trace/--store, 문제 해결)
- [brick-agent-plan.md](brick-agent-plan.md) — 개발 계획 (Phase 1–8)
- `.okf/00_agent_model/events.yaml` — canonical 이벤트 어휘
- `.okf/00_agent_model/behaviors.spec.yaml` — 통합 행동 스펙
- `.okf/00_agent_model/concepts/observability.md` — 관측 가능성 명세
- [ActiveGraph](https://activegraph.ai) · [OKF Skills](https://github.com/scaccogatto/okf-skills)

---

## 📄 라이선스

MIT License (배포 전 `LICENSE` 파일 추가 예정 — Phase 8)
