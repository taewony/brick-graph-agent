---
type: concept
title: "Brick 01: Event-Driven Injestor"
status: draft
trust_tier: unverified
provenance: "ActiveGraph 1.10.0 Event Store 명세"
updated_at: 2026-08-05
references:
  - brick_02_compiler.md
---
# Brick 01: 지식 수집 레이어
* **역할**: 로컬 또는 API로 유입되는 마크다운 지식을 파싱합니다.
* **ActiveGraph 이벤트**: `okf/knowledge/ingested` 이벤트를 영구 로그 스트림에 발행합니다.