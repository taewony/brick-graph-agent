---
type: AtomicConcept
id: atomic.cache_hit_detection
title: Cache Hit Detection (캐시 히트 탐지)
description: "프리픽스 캐시에서 저장된 프리픽스와 현재 요청 프리픽스를 비교해 히트·미스를 판단하고, 필요한 경우 캐시를 업데이트한다."
status: draft
generated:
  by: agent:builder/1.0
  at: 2026-08-06T08:07:41Z
verified: []
---

# Cache Hit Detection (캐시 히트 탐지)

## 📌 개념 정의
**Cache Hit Detection**은 요청이 들어올 때 **프리픽스**를 해시화하고 기존 캐시와 매칭시켜 **히트 여부**를
판단한다.
- **히트** → 저장된 KV/hidden 상태를 로드하고 바로 진행.
- **미스** → 프리패딩을 수행하고 새 프리픽스와 결과를 캐시에 삽입.

## ⚙️ 핵심 알고리즘
```pseudo
function detect_hit(request_prefix):
    h = hash(request_prefix)
    if h in cache_table:
        return (True, cache_table[h])
    else:
        return (False, None)

• 해시 충돌 방지: SHA‑256 기반 해시와 충돌 시 검증 단계(실제 토큰 비교) 수행.
## 🔗 관련 관계

• PREREQUISITES: prefix_cache
• COMPOSED_OF: caching_strategy (전략 선택)