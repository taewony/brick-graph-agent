---
type: AtomicConcept
id: atomic.distributed_kv
title: Distributed KV (분산 키‑값 저장소)
description: 여러 노드에 KV 파티션을 저장·복제해 확장성·내결함성을 제공하고, 로드 밸런서를 통해 요청을 라우팅한다.
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T08:11:19Z
verified: []
---

# Distributed KV (분산 키‑값 저장소)

## 📌 개념 정의
**Distributed KV**는 **키‑값** 데이터를 **여러 서버 인스턴스**에 **샤딩(파티셔닝)**하고 **복제**하여
- **수평 확장**(수평 스케일링)
- **고가용성**(HA) 및 **내결함성**(fault‑tolerance)

을 제공한다. 각 노드는 독립적인 KV 엔진(예: Redis, RocksDB) 을 실행하고, 메타데이터 서비스가 파티션 위치와 복제
상태를 관리한다.

## ⚙️ 핵심 구성 요소
| 컴포넌트 | 역할 |
|----------|------|
| **Partitioner** | 입력 키를 해시하고, `hash(key) % N` 로 파티션(샤드) 번호 결정 |
| **Replica Manager** | 각 파티션을 1+개의 복제본에 저장, 리더‑팔로워 동기화 |
| **Meta Service** | 파티션‑노드 매핑, 리더 선출, 헬스 체크 정보를 중앙 관리 |
| **Consistency Layer** | 읽기/쓰기 일관성 모델 제공 (Strong, Eventual, Read‑After‑Write 등) |
| **Load Balancer** | 클라이언트 요청을 적절한 KV 인스턴스로 라우팅 (다음 파일 `load_balancer.md` 참고) |
| **Fault Tolerance** | 노드 장애 감지 → 자동 리플리케이션 재배치 (다음 파일 `fault_tolerance.md` 참고) |

## 🚀 주요 동작 흐름
1. **쓰기**
   - 클라이언트 → Load Balancer → `Partitioner` 로 파티션 결정
   - 메타 서비스가 해당 파티션의 **리더** 노드(Primary) 식별
   - 리더에 쓰기 → 복제본(Secondary)에게 **동기식/비동기식** 복제 전파
2. **읽기**
   - 클라이언트 → Load Balancer → `Partitioner` 로 파티션 결정
   - 메타 서비스가 **리더** 혹은 **팔로워** 중 일관성 요구에 맞는 노드 선택
   - 선택된 노드에서 데이터 반환
3. **노드 장애**
   - 헬스 체크 실패 → `Fault Tolerance` 가 새로운 리더 선출 및 복제 재배치
   - 메타 서비스가 파티션‑노드 매핑을 업데이트하고 Load Balancer에 전파

## 🔗 관련 관계
- **PREREQUISITES**: `kv_cache` (키‑값 캐시와 연계)
- **COMPOSED_OF**: `load_balancer`, `fault_tolerance`, `distributed_serving` (전반적인 서비스 레이어)