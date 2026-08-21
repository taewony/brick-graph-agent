---
type: AtomicConcept
id: atomic.load_balancer
title: Load Balancer (로드 밸런서)
description: "요청을 여러 KV 서버 인스턴스에 고르게 분산시켜 처리량을 향상하고, 지연 시간을 최소화하는 프론트엔드 컴포넌트."
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T08:11:19Z
verified: []
sources:
  - id: vllm-paper
    resource: https://arxiv.org/abs/2209.06155
    title: vLLM - Easy, Fast and Cheap LLM Serving with PagedAttention
---

# Load Balancer (로드 밸런서)

## 📌 개념 정의
**Load Balancer**는 **클라이언트 요청**을 여러 **Distributed KV** 인스턴스로 라우팅하여 부하를 균등하게 분산한다.
주요 방식:
- **라운드‑로빈 (Round‑Robin)**
- **가중 라운드‑로빈 (Weighted RR)** – 인스턴스 성능에 따라 가중치 부여
- **최소 연결 (Least Connections)** – 현재 연결 수가 가장 적은 인스턴스로 전달
- **헬스 체크** – 비활성 인스턴스 자동 제외

## ⚙️ 핵심 흐름
1. **헬스 체크** – 주기적으로 각 KV 노드 상태 확인 (`ping`, `heartbeat`).
2. **라우팅 정책 적용** – 라우팅 알고리즘에 따라 대상 노드 선택.
3. **프록시 전달** – 선택된 노드에 HTTP/gRPC/JSON‑RPC 등 프로토콜로 요청 전달.
4. **응답 반환** – 노드 응답을 클라이언트에 바로 전달.

## 🔗 관련 관계
- **PREREQUISITES**: [`distributed_kv`](distributed_kv.md)
- **PREREQUISITE_OF**: [`fault_tolerance`](fault_tolerance.md) (장애 시 재시도·전환)