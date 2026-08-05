---
type: concept
title: "Brick 02: Dynamic Graph Compiler"
status: draft
trust_tier: unverified
provenance: "ActiveGraph Reactive State Machine"
updated_at: 2026-08-05
references:
  - brick_03_ask.md
  - brick_04_browse.md
---
# Brick 02: 실시간 지식 그래프 컴파일 레이어
* **역할**: Injest 이벤트를 구독하여 실시간으로 메모리 내 노드와 엣지 가계도를 형상화합니다.
* **ActiveGraph 이벤트**: 상태 변동 시 `okf/graph/compiled` 이벤트를 브로드캐스트합니다.