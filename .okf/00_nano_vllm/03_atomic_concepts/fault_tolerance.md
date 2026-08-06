---
type: AtomicConcept
id: atomic.fault_tolerance
title: Fault Tolerance (장애 내성)
description: 분산 KV 및 서비스 인프라에서 노드·네트워크 장애 발생 시 자동 복구·재배치를 제공하여 서비스 가용성을
유지한다.
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T08:39:47Z
verified: []
---

# Fault Tolerance (장애 내성)

## 📌 개념 정의
**Fault Tolerance**는 **노드 장애**, **네트워크 파편화**, **서버 재시작** 등 다양한 오류 상황에서도 **분산 KV
시스템**이 지속적으로 동작하도록 보장하는 메커니즘이다. 핵심 목표는
1. **서비스 중단 최소화** (Zero‑downtime)
2. **데이터 손실 방지** (Strong durability)
3. **자동 복구** (Self‑healing)

## ⚙️ 주요 구성 요소
| 구성 요소 | 역할 |
|-----------|------|
| **Health Checker** | 각 KV 인스턴스에 주기적인 `ping/heartbeat` 전송, 응답 지연·실패를 감지 |
| **Leader Election** | 장애 발생 시 새로운 리더(Primary) 선출 – Raft, ZooKeeper, etcd 등 사용 |
| **Replica Rebalancer** | 복제본 수가 부족해지면 다른 정상 노드에 새 복제본을 생성 |
| **Failover Router** | Load Balancer가 장애 노드 대신 새로운 리더/복제본으로 트래픽 전환 |
| **Consensus Log** | 장애 전후 상태 변화를 로그에 기록해 일관성 및 복구 시점 결정 |
| **Graceful Degradation** | 일시적 장애 시 읽기‑전용 모드 혹은 캐시‑우선 모드 전환 |

## 🚀 동작 흐름
1. **헬스 체크** – `Health Checker` 가 노드 응답을 모니터링.
2. **장애 탐지** – 연속된 타임아웃 또는 오류율 초과 시 “장애”로 판단.
3. **리더 재선출** – 현재 파티션의 리더가 다운되면 `Leader Election` 알고리즘이 새로운 리더를 선정한다.
4. **복제 재배치** – `Replica Rebalancer` 가 부족한 복제본을 다른 가용 노드에 복제한다.
5. **트래픽 전환** – `Failover Router` 가 로드 밸런서 설정을 업데이트해 요청을 새로운 리더/복제본으로 라우팅한다.
6. **복구 완료** – 복구가 종료되면 `Consensus Log` 에 상태를 기록하고 정상 모드(`Read‑Write`) 로 복귀한다.

## 📐 일관성 옵션
- **Strong Consistency** – 리더 재선출 후, 새로운 리더와 기존 복제본 모두가 최신 로그를 확보할 때까지 쓰기를 일시
중단.
- **Eventual Consistency** – 장애 중에도 읽기·쓰기를 허용하고, 복구 시 백그라운드 동기화 수행.

## 🔗 관련 관계
- **PREREQUISITES**: `distributed_kv`, `load_balancer`
- **COMPOSED_OF**: `distributed_serving` (전체 서비스 레이어에서 장애 내성을 활용)