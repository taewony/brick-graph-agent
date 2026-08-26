`Green Contexts`(혹은 유사한 SM 파티셔닝 기능)는 GPU의 스트리밍 멀티프로세서(SM)를 **여러 개의 독립된 실행 환경(컨텍스트)으로 분할**하고, 각각에 별도의 CUDA 스트림을 바인딩하여, 서로 다른 커널이 **간섭 없이 동시에 실행**되도록 하는 기능입니다.  
이를 LLM-from-scratch 추론에 적용할 수 있는지, 그리고 어떻게 최적화할 수 있는지 분석해보겠습니다.

---

## ✅ Green Contexts의 핵심 아이디어

- 하나의 GPU를 여러 개의 **논리적 파티션**으로 나눕니다.
- 각 파티션은 자신만의 **CUDA 컨텍스트**와 **스트림**을 가지며, 할당된 SM만 사용합니다.
- 이를 통해 **긴 배치의 처리량(throughput) 중심 커널**과 **짧은 지연 시간(latency) 민감 커널**이 같은 프로세스 안에서도 물리적으로 SM 리소스를 두고 경쟁하지 않게 됩니다.

---

## 🧠 LLM 추론에서의 활용 포인트

### 1. Prefill(처리량) vs Decode(지연시간) 분리
LLM 추론 서빙 환경(Continuous Batching)에서는 **Prefill(새로운 프롬프트 처리)**과 **Decode(토큰 생성)**가 **동시에 진행**됩니다.

| 단계 | 특징 | 요구 사항 |
|------|------|-----------|
| Prefill | 512~수천 토큰, 연산량 큼, 긴 커널 | SM 전체를 활용해 높은 처리량 |
| Decode | 1 토큰, 연산량 작음, 짧은 커널 | 지연 시간을 수백 μs 이내로 유지해야 함 |

일반적인 실행에서는 Prefill 커널이 모든 SM을 점유하면, Decode 커널이 **실행 기회를 빼앗겨 지연**됩니다.  
**Green Contexts**를 적용하면:

- **Partition A (예: SM 0~39)**: Prefill 전용 컨텍스트. 처리량 극대화.
- **Partition B (예: SM 40~55)**: Decode 전용 컨텍스트. 지연 시간 보장.

이렇게 분리하면, **Prefill이 아무리 무거워도 Decode 커널은 자신의 SM에서 즉시 실행**될 수 있어, 사용자 체감 응답속도(TTFT, TPOT)를 안정적으로 유지할 수 있습니다.

### 2. cuTile 커널의 배치
cuTile로 작성한 Prefill용 FMHA 커널과, Decode용 GEMV/Fused 커널을 **각 컨텍스트에 연결된 스트림으로 런치**하면 됩니다.

```python
# 가상의 Green Contexts API (cuda.core에 포함될 것으로 예상)
ctx_prefill = cuda.core.Context.create_with_sm_mask(sm_mask=0xFFFFF)  # 대부분 SM
ctx_decode  = cuda.core.Context.create_with_sm_mask(sm_mask=0x00FFF)  # 일부 SM만

stream_prefill = ctx_prefill.create_stream()
stream_decode  = ctx_decode.create_stream()

# cuTile 커널 런치 시 stream 지정 가능 (ct.launch에 stream 파라미터 가정)
ct.launch(prefill_kernel, ..., stream=stream_prefill)
ct.launch(decode_kernel, ..., stream=stream_decode)
```

두 스트림은 **독립적인 SM 집합에서 동시에 실행**되므로, 완전한 병렬성이 확보됩니다.

### 3. 단일 요청 처리 시에도 도움될까?
LLM-from-scratch를 단일 클라이언트 추론(한 문장씩 처리)에 사용한다면, Prefill과 Decode가 **순차적으로** 실행되므로 Green Contexts의 이점이 거의 없습니다.  
하지만 다음과 같은 시나리오에서는 유용합니다:

- **여러 요청을 배치로 묶어 처리**하는 서빙 구현체를 함께 개발할 경우.
- **Prefill 전용과 Decode 전용 모델을 병렬로 로딩**하는 극단적 최적화.
- **추론과 동시에 다른 GPU 작업(예: KV 캐시 전처리, 통계 연산)이 있다면** 그 작업을 별도 파티션에 격리.

---

## 🔧 구체적 구현 전략 (cuda.core + cuTile)

NVIDIA CUDA 13.3 블로그에서 암시된 `cuda.core`의 새로운 Context API는 아마도 SM 파티셔닝을 직접 지원할 것입니다.  
이를 LLM 추론 루프에 통합하는 방법은 다음과 같습니다:

1. **초기화**  
   ```python
   from cuda.core.experimental import Context, Stream

   # Prefill 컨텍스트: 전체 SM 중 대부분 사용 (예: 112개 중 96개)
   ctx_pf = Context.create(num_sms=96)
   stream_pf = ctx_pf.create_stream()

   # Decode 컨텍스트: 남은 SM (예: 16개)
   ctx_dc = Context.create(num_sms=16)
   stream_dc = ctx_dc.create_stream()
   ```

2. **커널 준비**  
   cuTile로 미리 컴파일해둔 Prefill용 Fused Attention+FFN 커널과 Decode용 Fused Attention+FFN 커널을 각 컨텍스트에 로드해야 할 수도 있습니다.  
   다행히 cuTile의 JIT 컴파일은 일반적으로 컨텍스트 독립적이며, 생성된 CUDA 모듈을 여러 컨텍스트에서 사용할 수 있습니다.  
   만약 안 된다면, 각 컨텍스트에서 `ct.launch`를 호출해 따로 로드해야 합니다.

3. **서빙 루프 (Continuous Batching)**  
   ```python
   # Prefill 배치 준비 (여러 새 요청의 프롬프트를 패딩)
   # Decode 배치 준비 (기존 요청의 마지막 토큰)
   # 비동기로 두 커널을 각 스트림에 런치
   ct.launch(prefill_fused_kernel, ..., stream=stream_pf)
   ct.launch(decode_fused_kernel, ..., stream=stream_dc)

   # 두 스트림 동기화 없이 각각 독립적으로 완료됨
   stream_pf.synchronize()
   stream_dc.synchronize()
   ```

4. **주의사항**  
   - KV Cache는 두 파티션 간에 공유되어야 하므로, **UVA(Unified Virtual Addressing) 상에서 동일한 메모리 포인터**를 사용합니다. SM 파티셔닝은 실행 유닛만 분리할 뿐 메모리는 공유되기 때문에 문제 없습니다.
   - GPU 메모리 대역폭은 여전히 공유되므로, Prefill이 메모리 대역폭을 많이 사용하면 Decode 성능이 미세하게 영향을 받을 수 있습니다. 하지만 이는 어쩔 수 없는 물리적 한계이며, Green Contexts가 없을 때보다 훨씬 예측 가능해집니다.

---

## 📊 기대 효과

- **Decode latency 안정화**: Prefill 폭주에도 Decode SM은 독립적으로 작동하므로, P99 latency가 크게 개선됩니다.
- **처리량 증가**: Prefill에 더 많은 SM을 몰아줄 수 있어, 프롬프트 처리량이 증가하고 전체 서빙 효율이 오릅니다.
- **Python 런치 오버헤드와 무관**: CUDA Graph와 달리 런치 오버헤드 자체를 줄이지는 못하지만, SM 경쟁을 근본적으로 제거하여 실질적 성능을 높입니다.

---

## 🧩 결론

**Green Contexts는 LLM-from-scratch 추론 최적화에 충분히 적용 가능하며, 특히 다중 요청 서빙 시 Prefill과 Decode 간의 간섭을 없애는 강력한 수단입니다.**  
cuTile 커널을 각 컨텍스트에 맞게 스트림으로 분배하는 코드는 `cuda.core` API가 공개되면 비교적 쉽게 작성할 수 있습니다.  
단일 요청 처리만 한다면 효과가 제한적이지만, 확장성 있는 서빙 시스템을 구축할 계획이라면 **CUDA Graph + Green Contexts + cuTile 퓨전**이라는 3박자를 모두 갖춘 최고 수준의 최적화가 가능할 것입니다.

필요하시면 `cuda.core`의 예상 API를 바탕으로 더 구체적인 의사 코드를 작성해 드릴 수 있습니다.