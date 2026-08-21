# brick-agent 실증 논문 초고 (Empirical Paper Draft) — 구현 근거 기반

> 본 문서는 `papers/brick-agent-paper-01.md`(개념 골격)을 잇는 **실증 논문의 초고**입니다.
> 차이점: paper-01이 "무엇을 만들고자 하는가"라면, 본 문서는 **지금까지 실제로 구현·검증한
> brick-agent를 근거로** (1) 논문 초고가 어떻게 구성되는지, (2) 논문 수준의 실증을 위해
> **추가로 무엇을 개발·측정해야 하는지**를 정리합니다. 현재 측정값은 결정론적
> 시뮬레이션/단위·통합 테스트에 기반하며, 실제 LLM 벤치마크 수치는 "실험 프로토콜"로
> 명시적으로 분리했습니다.

---

## 0. 요약 (TL;DR)

지금까지 구현된 brick-agent는 하나의 구체적이고 **실증 가능한** 시스템이며, 논문의
핵심 기여는 paper-01의 "논리 모순 검출"보다 **다음 네 가지로 재정의**하는 것이
근거가 명확합니다.

1. **이벤트 소싱 기반 LLM 관측 + 결정적 재생** — 모든 LLM 호출을
   `llm.requested/llm.responded`(model·prompt_hash·answer·latency·cost)로 로그에 남기고,
   `ReplayReader`가 기록된 답변을 **재호출 없이** 서빙. (가장 강하게 구현된 신규성)
2. **구조적 지식 무결성 감사** — 8개 결정론적 그래프 디텍터가 실제 KB에서
   순환·dangling·단방향 관계·산문 노이즈·증빙 부재 **106건을 검출→수리(0건)**.
3. **regime→seam 자기 개선 루프의 구조 지식 확장** — mentor 루프가 OKF/SQL 타겟을
   구동하며, 실패 regime(예: concept-orphan) → prompt-transform 승격을 held-out 게이트로 검증.
4. **단일 런타임 위의 이중 서비스 에이전트** — OKF 지식 Q&A + 자연어 DB Q&A가
   동일 이벤트 소싱 기반(router + 공유 서비스 + 관측 계층)을 공유.

---

## 1. [Abstract] 초안

> Knowledge-grounded agents are rarely *auditable*: the LLM round-trips that produce an
> answer are invisible, so a wrong or costly answer cannot be reconstructed after the fact.
> We present **brick-agent**, an event-sourced dual-service agent — an OKF (Open Knowledge
> Format) Q&A service and a natural-language-to-SQL service — whose every model call is
> recorded as `llm.requested` / `llm.responded` events carrying the prompt hash, the recorded
> answer, model, latency, and cost. The append-only log doubles as an LLM observability store,
> and a content-keyed `ReplayReader` serves recorded answers so held-out validation replays
> **without re-invoking the model**. On top of this substrate, eight deterministic graph
> detectors audit the knowledge base for structural integrity (concept cycles, dangling
> references, one-sided relationship declarations, prose noise, and missing evidence) — they
> surfaced 106 issues in a real 54-document knowledge bundle and drove a repair pass to zero.
> A self-improving mentor loop classifies failures into regimes and promotes prompt transforms
> through a held-out gate, closing the loop from diagnosis to deployment with full audit
> trails. We present the system, the (deterministic) preliminary validation, and a full
> experimental protocol for measuring grounded-answer accuracy, NL2SQL execution accuracy,
> cost/latency, and improvement-over-rounds against flat-RAG and no-transform baselines.

---

## 2. [Introduction] — 문제 정의 (구현 기준으로 재정의)

paper-01은 "문서의 논리 모순 검출 + 계층적 학습"을 앞세웠지만, **현재 brick-agent의
실측 가능한 강점**은 그보다 아래 세 가지 병목입니다. 논문의 서론은 여기서 출발해야
실험이 따라옵니다.

- **B1. LLM 관측 불가능성** — 실무 RAG/agent 시스템에서 "이 답변이 왜, 어떤 비용으로,
  어떤 프롬프트로 나왔는가"를 사후 복원할 수 없다. 우리는 이 문제를 **이벤트 소싱**으로
  해결: 호출 자체가 append-only 로그의 일부가 된다.
- **B2. 지식 기반의 무결성 무감지** — 기존 KB 린터(frontmatter/링크 검사)는 본문 관계
  블록의 순환·dangling·단방향 기재를 **못 본다**. 우리는 그래프 구조 위에서 8개 디텍터로
  이를 잡고, 실제 번들에서 106건을 찾아 수리했다.
- **B3. 개선의 신뢰성** — 자동 개선 루프는 "왜 승격했는가"를 증명해야 한다. 우리는
  Regimes의 regime→seam 매핑을 계승하되, 개선 대상이 프롬프트뿐 아니라 **지식 구조
  무결성 신호**까지 포함되도록 확장하고, 검증을 결정적 재생으로 고정한다.

**기여 요약**: (i) prompt-hash 기반 결정적 재생을 갖춘 이벤트 소싱 LLM 관측 계층,
(ii) 구조적 지식 무결성을 위한 8-디텍터 감사 + 재현 가능한 복구 파이프라인,
(iii) regime→seam 자기 개선 루프의 OKF/NL2SQL 이중 타겟 통합,
(iv) 위 모두를 단일 CLI(`brick-agent`)로 노출한 엔드투엔드 시스템.

---

## 3. [System] — 실제 구현 (측정 가능한 설계)

### 3.1 구성 요소 (모두 구현 완료, 73 tests)

| 계층 | 모듈 | 실증 관점의 역할 |
|---|---|---|
| OKF 타겟 | `src/core/targets/okf/` (outcome·taxonomy·action_space·prompt_transforms·eval·target) | grounded Q&A의 근거/판정/lint |
| OKF 런타임 | `src/agents/okf/` (ingest/lint/ask 체인) | 이벤트 체인으로 KB 적재·감사·답변 |
| SQL 타겟 | `src/core/targets/sql/` | 자연어→SQL (스키마 인코딩→컬럼 스코어링→드래프트) |
| 공유 서비스 | `src/runtime/` (reader_registry·embedder·guardrails·history_logger·loader·event_store·observability·replay) | 관측·재생·가드레일 |
| Mentor 루프 | `src/core/loop/` + 양 타겟의 taxonomy/action_space | baseline→diagnose→gates→promote→attribute |
| CLI/라우터 | `src/agents/{brick_agent,router}.py` | 단일 진입점, 역할 자동 분류 |

### 3.2 핵심 메커니즘 (논문 Figure/Table 대상)

1. **LLM 관측 시임**: `ask_with_observability(graph, reader, ...)`가
   `llm.requested`(prompt_hash=sha256(prompt)) → `call_reader`(타이밍/에러 캡처) →
   `llm.responded`(caused_by, model, answer, latency, error, cost?)를 발행.
2. **결정적 재생**: `build_replay_cache(events)`가 `llm.responded.caused_by → llm.requested`
   쌍에서 `prompt_hash → answer`를 하베스트하고, `ReplayReader`가 캐시 히트 시
   **fallback reader를 호출하지 않음**(fallback이 실패하도록 한 테스트로 입증).
3. **영속**: `Runtime(graph, persist_to=...)`로 `data/events.db`(SQLite)에 run별 append-only;
   `Runtime.load/fork` + `replay_into_graph`로 그래프 재구성.
4. **감사 디텍터 8종**: `cyclic_concept`, `inverse_relationship`, `dangling_reference`,
   `relationship_noise`, `concept_orphan`, `rule_schema_reference`, `ambiguous_trigger`,
   `missing_evidence` — 각각 `{code, node, detail}` 신호를 반환.
5. **복구 파이프라인**: `src/tools/okf_repair.py`가 body 교정(순환 방향 수정·dangling 재지정·
   노이즈 제거) + 증빙 추가 + 단방향 관계 미러링을 멱등하게 수행하고, `--linkify`가
   module→composite→atomic 하향 트리를 마크다운 링크로 연결.
6. **지식 브라우저**: `browse`가 `docs/<kb>/index.html` 대시보드(vis-network + 히스토리
   네비게이션 + 상대경로 정규화)를 생성.

---

## 4. [Preliminary Validation] — 이미 검증된 것 (결정론적)

> 아래 수치는 실 LLM 벤치마크가 아니라, 구현·단위·통합 테스트와 결정론적 감사에서
> 얻은 **예비 검증**이다. 논문에서 "Preliminary (deterministic)"로 명확히 표기한다.

| 검증 항목 | 결과 | 근거 |
|---|---|---|
| 시스템 정합성 | 73/73 tests 통과 (OKF/SQL/mentor/관측/재생/CLI/KB 감사) | `pytest tests/` |
| KB 무결성 감사 | 실제 54문서 번들에서 **106건 검출 → 0건 수리** (순환 7·단방향 53·dangling 14·노이즈 4·증빙 28) | `lint_knowledge_graph` |
| KB 재검증 | `okf_validate.py` 0 errors (수리 후) | OKF v0.2 validator |
| 결정적 재생 | 저장 run의 이벤트 멀티셋이 재생 그래프와 동일 + `ReplayReader`가 **LLM 재호출 0회**로 답변 서빙 | `tests/test_phase5_observability.py`, `test_phase8_integration.py` |
| Mentor 자기 개선(시뮬) | SQL·OKF 실패 시리즈에서 regime 분류→transform 초안→gate→승격→**다음 run이 개선본 사용** (정확도 0→1.0, 동일 시드) | `tests/test_mentor_loop.py` |
| 관측 집계 | run별 LLM 호출 수·평균 지연·에러 집계 (`mentor status`) | `observability.summarize_llm_usage` |
| 그래프 트리 정비 | module→composite→atomic 하향 링크가 마크다운 링크로 연결, 대시보드 엣지 92→325 | `okf_repair --linkify` + `browse` |

**의미**: "관측 가능·재생 가능·감사 가능"이라는 시스템적 주장은 **이미 검증**되었고,
남은 것은 (아래 §5) **정확도/비용/학습효율의 실측**이다.

---

## 5. [Experimental Protocol] — 실증 평가 설계 (RQs & 메트릭)

논문의 핵심 실험은 다음 네 연구질문(RQ)으로 구성한다.

### RQ1. grounded Q&A 정확도 — 지식 구조가 답변 품질을 올리는가?
- **데이터**: `01_nano_vllm`(54문서)에서 표본 추출한 질문 100~300개 + 정답/근거 개념 라벨.
  사내 문서 시나리오로 확장 가능(OKF 번들로 수집·정규화).
- **조건**: (a) brick-agent(개념 트리 확장 + 규칙 주입 + trim), (b) flat-RAG(개념 트리/규칙
  없이 임베딩 top-k 본문만), (c) no-transform(build_prompt_parts 파이프라인 제거).
- **메트릭**: 정답 판정 정확도(LLM judge/정규화 매치), 근거 개념 재현율
  (answer가 gold 개념 id를 인용하는 비율), `applied_transforms`별 절제.
- **가설**: (a) > (b)·(c), 특히 concept-orphan/ambiguous-trigger가 많은 질문에서 격차.

### RQ2. NL2SQL 실행 정확도
- **데이터**: `data/store_front.db`(5테이블·FK) 위 50~100개 자연어 질문+gold SQL; 확장 시
  Spider-lite 등 공개 벤치마크.
- **메트릭**: execution accuracy(결과집합 동일), 논리형태 정확도(구조 일치).
- **조건**: brick-agent vs. 스키마 나열 프롬프트(컬럼 스코어링 제거).

### RQ3. 비용·지연 관측의 정밀도
- `--real`(AnthropicReader)로 동일 질문 N회 실행, `llm.responded`의 latency/cost/usage가
  실제 API 사용량과 일치하는지, `ReplayReader`가 동일 프롬프트에서 비용 0으로 재생하는지
  측정. **관측 계층은 구현됐으나 실측 미실행** — 핵심 추가 개발 항목.

### RQ4. mentor 루프의 실제 개선 (hold-out)
- 실제 reader + (가능하면 LLM author)로 실패 질문 집합에 대해 N라운드
  `run_loop` 수행. 메트릭: 라운드별 hold-out 정확도 궤적, promotion/discard 퍼널,
  over-promotion 비율, **McNemar + bootstrap CI**, wall(개선 불가 regime) 명명.
- **통계**: Regimes 방식을 차용 — OPTIMIZE/CONFIRM 분할, 5 시드, `overall_delta`,
  `target_delta`, per-type 회귀 검사.

### 공통
- **베이스라인**: flat-RAG, no-transform, prompt-only LLM.
- **평가 환경**: 동일 event store에 기록해 재현 가능하게; 모든 결과를 이벤트 로그로 배포.

---

## 6. [Expected Results & Threats to Validity]

- **기대**: RQ1/RQ2에서 grounded 조건이 정확도·근거율 우위, RQ3에서 관측 정밀도와
  결정적 재생의 비용 절감, RQ4에서 hold-out 게이트가 과적합 승격을 걸러냄.
- **위협**: (i) 소규모 단일 KB → 일반화 한계(다중 번들로 완화), (ii) LLM judge의 비결정성 →
  결정적 재생/고정 judge로 완화, (iii) mock-mode 결과의 외적 타당성 → 실측과 분리 표기,
  (iv) 한국어/도메인 특화 프롬프트 편향.

---

## 7. [추가 개발 필요] — 실증 논문으로 만들기 위한 작업 (우선순위)

아래가 "지금까지 구현을 넘어 논문 실험을 실행하려면" 추가로 만들어야 할 것들입니다.
**P0가 없으면 실측 수치가 나오지 않습니다.**

| 우선순위 | 작업 | 산출물 |
|---|---|---|
| **P0-1** | 실 LLM 평가 하네스: 질문셋(정답+근거 라벨) 로더, `--real` 실행, LLM-judge/정규화 판정, 결과를 이벤트 로그로 저장 | RQ1/RQ2 정확도 수치 |
| **P0-2** | cost/latency 실측: `AnthropicReader.answer`의 usage/cost를 `llm.responded`에 기록, `summarize_llm_usage`를 토큰 단위로 확장 | RQ3 수치 |
| **P0-3** | NL2SQL 벤치마크: store_front 파생 gold 질문셋 + Spider-lite 어댑터 | RQ2 수치 |
| **P1-1** | LLM author(실 개선안 초안)로 mentor 루프 실험 실행: `LLMSqlAuthor`/OKF용 author, 라운드 궤적·퍼널·McNemar 스크립트 | RQ4 수치 |
| **P1-2** | 비교 베이스라인: flat-RAG / no-transform / prompt-only 구현을 action_space 옵션으로 | RQ1/RQ2 조건 |
| **P1-3** | guardrail 실측: `required_lint_before_ask`를 ask 체인에 강제하고, 강제 전후 정확도/에러율 측정 | 시스템 신뢰성 증거 |
| **P2-1** | 논리 모순 검출(paper-01의 헤드라인): Claim/Evidence 노드 + SUPPORTS/CONTRADICTS 디텍터를 문서 코퍼스에 적용(현재는 concept/rule/schema 중심) | paper-01 주장의 실증 |
| **P2-2** | 학습 효율 사용자 스터디: `browse` 대시보드 vs 평문 문서로 개념 숙지 시간·퀴즈 정확도 측정 | paper-01의 "학습 25% 단축" 주장 실측 |
| **P2-3** | 다중 도메인 확장: `02_store_front`·`00_agent_model`·사내 번들을 평가에 포함해 일반화 검증 | 외적 타당성 |
| **P2-4** | 통계·재현 패키지: 실험 시드/스플릿 고정, `Makefile`에 `experiments/*` 스크립트, 결과 테이블 생성 | 논문 표 재현 |

---

## 8. [Conclusion] 초안

> brick-agent shows that **observability and auditability can be properties of the substrate,
> not add-ons to the application**. Recording every LLM round-trip as an event — and making
> validation replay from recorded answers without a fresh model call — turns a knowledge
> agent into a measurable, reproducible system. The eight structural-integrity detectors
> demonstrate that a graph substrate can turn "knowledge drift" from an invisible risk into
> a countable, fixable signal: 106 issues found and repaired on a real bundle. With the
> evaluation protocol in §5 executed against real LLM baselines, we can close the loop from
> "deterministically auditable" to "empirically better" — the substrate is the lever, the
> log is the proof, and the held-out gate is the trust boundary.

---

## 9. 결론 (개발자 관점 요약)

- **논문의 실측 가능한 스토리는 "이벤트 소싱 LLM 관측 + 구조 지식 감사 + 결정적 재생"** 입니다.
  paper-01의 "논리 모순 검출"은 현 구현에서 아직 부재하므로, 실측 전에 P2-1로 보강하거나
  논문 범위에서 명시적으로 제외해야 합니다.
- **지금 당장 쓸 수 있는 수치**는 73 tests·106→0 감사·결정적 재생·시뮬 mentor 승격입니다.
  이들을 "Preliminary (deterministic)"로 표기하고, RQ1–RQ4를 "to be executed"로 두는 것이
  정직한 초고입니다.
- **다음 개발 우선순위는 P0(실 LLM 평가 하네스 + cost/latency 실측 + NL2SQL 벤치마크)** —
  이것만 붙으면 실측 테이블이 나오고, 논문 draft → 실측 논문으로 전환됩니다.
