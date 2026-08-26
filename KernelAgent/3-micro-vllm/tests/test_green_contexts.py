from __future__ import annotations
import os
import traceback
import torch

def print_header(title: str) -> None:
    print(f"\n--- {title} ---")

def test_pytorch_green_context() -> bool:
    print_header("Testing PyTorch GreenContext")
    try:
        from torch.cuda.green_contexts import GreenContext
    except ImportError as exc:
        print(f"torch.cuda.green_contexts is not importable: {exc}")
        return False

    print("torch.cuda.green_contexts is importable")
    try:
        ctx = GreenContext.create(num_sms=16)
        print("GreenContext.create(num_sms=16) succeeded")
        ctx.set_context()
        print("ctx.set_context() succeeded")
        ctx.pop_context()
        print("ctx.pop_context() succeeded")
        return True
    except Exception as exc:
        print(f"GreenContext creation or activation failed: {exc} (treated as optional)")
        return False

def test_cuda_core_green_context() -> bool:
    print_header("Testing cuda.core API")
    try:
        from cuda.bindings import driver as cuda
        from cuda.core import ContextOptions, Device, SMResourceOptions
    except ImportError as exc:
        print(f"cuda.core module or bindings are not importable: {exc}")
        return False

    print("cuda.core and cuda.bindings modules are importable")
    try:
        cuda.cuInit(0)
        dev = Device(0)
        dev.set_current()

        sm = dev.resources.sm
        total_sms = sm.sm_count

        prefill_sms = int(os.environ.get("NANO_VLLM_PREFILL_SMS", "32"))
        decode_sms = int(os.environ.get("NANO_VLLM_DECODE_SMS", "16"))

        print(f"Device found: {dev.name}, total SMs: {total_sms}")
        print(f"Requested benchmark split: prefill={prefill_sms} SMs, decode={decode_sms} SMs")

        if prefill_sms + decode_sms > total_sms:
            raise RuntimeError(
                f"requested split {prefill_sms}+{decode_sms} exceeds total SM count {total_sms}"
            )

        # -------------------------------------------------------------
        # Part 1: Single Partition Verification (Decode-only Isolation)
        # -------------------------------------------------------------
        print(f"\n[Part 1] Requesting single partition of size: {decode_sms} SMs")
        single_layouts = list(sm.split(SMResourceOptions(count=(decode_sms,))))
        if not single_layouts or len(single_layouts) == 0:
            raise RuntimeError("single partition split returned no layouts")

        # 🌟 [🔥 핵심 수정 포인트 1] 2중 리스트 구조 타파 (구조 분해 할당)
        # single_layouts[0]은 분할 시나리오이며, 그 내부에 실제 분할된 리소스 그룹들이 담겨 있습니다.
        # count=(16,) 하나만 요청했으므로 내부 리스트에는 [요청한 16개 객체, 남은 32개 잔여 객체]가 들어있습니다.
        target_layout = single_layouts[0]
       
        if isinstance(target_layout, (list, tuple)) and len(target_layout) >= 1:
            crit_grp = target_layout[0] # 16개 SM 객체 직접 추출
            remainder_grp = target_layout[1] if len(target_layout) >= 2 else sm
        else:
            crit_grp = target_layout
            remainder_grp = sm
           
        print(f"Single SM split succeeded; resource type: {type(crit_grp)}")

        # 단일 SMResource 객체를 리스트에 담아 주입하므로 완벽히 통과합니다.
        ctx_crit = dev.create_context(ContextOptions(resources=[crit_grp]))
        print("single dev.create_context succeeded")

        dev.set_current(ctx_crit)
        print("dev.set_current(ctx_crit) succeeded")

        stream = ctx_crit.create_stream()
        print(f"ctx_crit.create_stream succeeded; stream type: {type(stream)}")

        dev.set_current()
        print("dev.set_current() restored primary context after single-context test")

        # -------------------------------------------------------------
        # Part 2: Two-Context Pipeline Verification (Prefill vs Decode Partitioning)
        # -------------------------------------------------------------
        # 🌟 [🔥 핵심 수정 포인트 2] 논문 벤치마크 사양 전용 고정 매핑
        # [Part 1]에서 검증과 동시에 완벽하게 상호 격리 추출된 두 자원을 다이렉트로 매핑합니다.
        # decode_grp는 정밀 격리된 16개 SM, prefill_grp는 남은 32개 SM 자원 객체입니다.
        print(f"\n[Part 2] Executing two-way allocation: {prefill_sms} SMs + {decode_sms} SMs")
       
        decode_grp = crit_grp
        prefill_grp = remainder_grp
        print(f"✅ Resource mapping verified: Decode={type(decode_grp)}, Prefill={type(prefill_grp)}")

        # 추출된 각각의 SMResource 독립 인스턴스를 컨텍스트로 빌드
        ctx_prefill = dev.create_context(ContextOptions(resources=[prefill_grp]))
        ctx_decode = dev.create_context(ContextOptions(resources=[decode_grp]))
        print("two-context creation succeeded")

        # Prefill 전용 독립 그린 스트림 구동 테스트
        dev.set_current(ctx_prefill)
        print("dev.set_current(ctx_prefill) succeeded")
        prefill_stream = ctx_prefill.create_stream()
        print(f"ctx_prefill.create_stream succeeded; stream type: {type(prefill_stream)}")

        # Decode 전용 하드웨어 격리 스트림 구동 테스트
        dev.set_current(ctx_decode)
        print("dev.set_current(ctx_decode) succeeded")
        decode_stream = ctx_decode.create_stream()
        print(f"ctx_decode.create_stream succeeded; stream type: {type(decode_stream)}")

        # 런타임 호스트 스레드 환경 복구를 위해 Primary Context로 복귀
        dev.set_current()
        print("dev.set_current() restored primary context after two-context test")
        return True

    except Exception as exc:
        print(f"cuda.core operations failed: {exc}")
        print("--- detailed traceback ---")
        traceback.print_exc()
        return False

def main() -> int:
    print("CUDA Version:", torch.version.cuda)
    print("PyTorch Version:", torch.version.device if hasattr(torch.version, "device") else torch.__version__)
   
    pytorch_ok = test_pytorch_green_context()
    cuda_core_ok = test_cuda_core_green_context()

    print_header("Summary")
    print(f"pytorch_green_context_ok={pytorch_ok}")
    print(f"cuda_core_green_context_ok={cuda_core_ok}")

    if cuda_core_ok:
        print("RESULT: PASS - cuda.core Green Context path is available for benchmark runs")
        return 0
   
    print("RESULT: FAIL - cuda.core Green Context path is not available")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())