---
type: concept
title: "Brick 03: Context-Aware Ask Engine"
status: draft
trust_tier: unverified
provenance: "LLM Context Window Engineering"
updated_at: 2026-08-05
references: []
---
# Brick 03: 지식 추론 레이어
* **역할**: 질문 인입 시 컴파일러에서 제공하는 연관 컨텍스트 서브그래프를 획득하여 RAG 프롬프트를 구성합니다.
* **ActiveGraph 이벤트**: `agent/query/processed` 이벤트를 남겨 추론 과정을 투명하게 기록합니다.