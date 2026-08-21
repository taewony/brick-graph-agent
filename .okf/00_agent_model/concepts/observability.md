---
type: concept
title: "brick.agent Observability Spec (Event-Sourcing 기반 LLM 관측)"
status: proposed
trust_tier: unverified
provenance: "activegraph 1.10.0 (EventStore / llm.* / trace.causal_chain / observability) 분석"
updated_at: 2026-08-14
references:
  - ../events.yaml
  - ../behaviors.spec.yaml
  - ../examples/brick.agent.model.md
  - ../history.yaml
---

# brick.agent Observability Spec

## 1. 목적

brick.agent는 **이벤트 소싱 구조** 위에서 LLM 호출의 관측 가능성(observability)과
에이전트 동작 가시성(visibility)을 확보한다. 이 문서는 그 계약을 명세한다:

- 모든 에이전트 행동은 append-only 이벤트 로그에 남는다 (이벤트 = 사실).
- **모든 LLM 왕복은 `llm.requested` / `llm.responded` 이벤트로 기록**되어
  모델·비용·지연·캐시·프롬프트 해시가 로그에서 복원 가능하다.
- 그래프 객체·답변·결정의 **인과 사슬**(causal chain)이 이벤트 로그만으로 재구성된다.

## 2. As-Is 분석 (2026-08-14)

| 항목 | 상태 |
|------|------|
| activegraph 1.10.0 이벤트 소싱 (`EventStore`, `replay_into`, `Runtime.load/fork`, SQLite/Postgres 백엔드) | 런타임이 네이티브 제공, **미연결** |
| `@llm_behavior` + `llm.requested/responded` (model/prompt_hash/cost_usd/cache_hit) | 런타임이 네이티브 제공, **미사용** |
| `trace/causal_chain` (객체 → LLM 왕복 → goal 인과 사슬) | 런타임 제공, **미사용** |
| `observability/` (구조화 로깅 JSON 스키마, OpenTelemetry, Prometheus) | 런타임 제공, **미사용** |
| SQL·OKF Reader 호출 (`_READERS` dict + `reader.answer()`) | 현재 구현 — **LLM 왕복이 로그에 기록되지 않는 갭** |
| 이벤트 어휘 | 3개 방언 혼재(예: `okf/knowledge/ingested` vs `okf.ingested` vs `GRAPH_COMPILED`) → `events.yaml`으로 단일화 완료 |
| KB 무결성 | Phase 2 lint가 `.okf/01_nano_vllm`에서 개념 순환 7건 검출 (예: `continuous_batching ↔ iteration_level_scheduling`) — lint 신호가 관측 대상 |

## 3. 계약 (Contract)

### 3.1 이벤트 계약
- 이벤트 이름/페이로드는 `events.yaml`(canonical)을 따른다.
- 모든 payload는 `request_id`(또는 `question_id`)와 `run_id`를 포함한다.
- 실패는 payload 필드(`error`/`issues`)로 기록한다 (예외가 아님).

### 3.2 LLM seam 계약 (최우선)
`sql_agent.draft_query`와 `okf_agent.generate_answer`는 `reader.answer()` 호출을
**반드시 `llm.requested` / `llm.responded` 이벤트로 감싼다**:

```python
# behaviors.py (규약 예시)
graph.emit(E.LLM_REQUESTED, {
    "request_id": request_id,
    "model": getattr(reader, "name", ""),
    "prompt": prompt,
    "prompt_hash": sha256(prompt.encode()).hexdigest(),
})
t0 = time.perf_counter()
try:
    answer = reader.answer(context=prompt, question=question, question_id=request_id)
    err = ""
except Exception as e:
    answer, err = "", f"{type(e).__name__}: {e}"
graph.emit(E.LLM_RESPONDED, {
    "request_id": request_id,
    "caused_by": <llm.requested event id>,
    "model": getattr(reader, "name", ""),
    "latency_seconds": time.perf_counter() - t0,
    "error": err,
})
```

이를 통해 로그가 곧 LLM 관측 저장소가 된다: 질문별 모델·비용·지연·캐시·실패 추적 가능.

### 3.3 인과 추적
- `activegraph.trace.causal_chain(graph, object_id)`를 사용해
  "답변 → 컨텍스트 → 개념/규칙 → LLM 왕복(model, cost) → 최초 요청" 사슬을 렌더링.
- `@llm_behavior`로 전환하면 객체 provenance에 `llm_request_event_id`가 자동 기록되어
  별도 작업 없이 위 사슬에 LLM 구간이 삽입된다 (장기 목표).

### 3.4 영속·재생·결정적 검증
- `Runtime(graph, store=SQLiteEventStore(path))` 또는 `persist_to=...`로 run 단위 영속.
- `Runtime.load` / `Runtime.fork`로 과거 run 재생·분기.
- `llm/cache.py`(prompt_hash 기반 응답 캐시)를 활성화하면 **held-out 검증 시
  LLM을 재호출하지 않는 결정적 재생**이 가능 → mentor의 candidate→validate→promote
  게이트(모델러 §12)가 실제로 동작.
- 재생 중 응답 불일치는 `ReplayDivergenceError`로 탐지.

### 3.5 로깅·메트릭
- `activegraph.observability.logging.configure_logging()` — JSON 라인 스키마
  (`run_id, event_id, behavior, model, cost_usd, latency_seconds, error_type...`)로 대시보드 연동.
- `OpenTelemetryMetrics` / Prometheus로 이벤트 볼륨·LLM 비용·실패율 집계.
- `brick_04_browse`(실시간 브라우저)는 이벤트 스트림 + `causal_chain`을 데이터 소스로 사용.

## 4. 이행 로드맵 (우선순위)

1. **[P0] LLM seam 전환** — `draft_query`/`generate_answer`에서 `llm.requested/responded` emit.
2. **[P0] 이벤트 어휘 단일화** — `events.yaml`을 canonical으로, 레거시 이름은 `aliases`로 매핑 (완료).
3. **[P1] 영속 + 재생 연결** — SQLiteEventStore + `Runtime.load/fork` + llm 캐시.
4. **[P1] 인과 대시보드** — `causal_chain` 기반 "왜 이 답변이 나왔나" 뷰 (brick_04).
5. **[P2] Mentor 관측 이벤트 소비** — `mentor.observe`가 correctness 외 cost/latency/실패 이벤트도 입력으로.

## 5. 성공 기준

- [ ] 모든 LLM 호출이 `llm.requested/responded`로 로그에 남는다 (모델·cost·latency·cache).
- [ ] 임의 답변에서 `causal_chain`으로 전체 인과 사슬을 렌더링할 수 있다.
- [ ] SQLite store로 run 재생이 가능하고, llm 캐시로 결정적 재실험이 가능하다.
- [ ] JSON 로그/메트릭으로 질문별 LLM 비용·지연·실패율을 집계할 수 있다.
