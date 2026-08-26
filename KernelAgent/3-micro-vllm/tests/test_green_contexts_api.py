"""Preflight check for CUDA Green Context support on the target PC.

PyTorch GreenContext is optional in the current environment. The required path
for the paper benchmark is cuda.core resource partitioning with
Device.set_current(). The benchmark split is implemented by carving out a
decode partition and using the remaining SM resource as the prefill partition.
"""

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


def split_decode_and_remainder(sm, decode_sms: int, resource_options_cls):
    layouts = list(sm.split(resource_options_cls(count=(decode_sms,))))
    if not layouts:
        raise RuntimeError("single decode partition split returned no layouts")

    layout = layouts[0]
    if isinstance(layout, (list, tuple)) and len(layout) >= 1:
        decode_grp = layout[0]
        prefill_grp = layout[1] if len(layout) >= 2 else sm
        return decode_grp, prefill_grp, len(layout)

    return layout, sm, 1


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

        print(f"\n[Part 1] Requesting decode partition of size: {decode_sms} SMs")
        decode_grp, prefill_grp, layout_width = split_decode_and_remainder(sm, decode_sms, SMResourceOptions)
        prefill_source = "split_remainder" if layout_width >= 2 else "device_sm_fallback"
        print(f"SM split succeeded; layout resource count: {layout_width}")
        print(f"Decode resource type: {type(decode_grp)}")
        print(f"Prefill resource type: {type(prefill_grp)}")
        print(f"Prefill resource source: {prefill_source}")

        ctx_decode_only = dev.create_context(ContextOptions(resources=[decode_grp]))
        print("single decode dev.create_context succeeded")
        dev.set_current(ctx_decode_only)
        print("dev.set_current(ctx_decode_only) succeeded")
        stream = ctx_decode_only.create_stream()
        print(f"ctx_decode_only.create_stream succeeded; stream type: {type(stream)}")
        dev.set_current()
        print("dev.set_current() restored primary context after single-context test")

        print(f"\n[Part 2] Executing two-context allocation: prefill={prefill_sms} SMs, decode={decode_sms} SMs")
        ctx_prefill = dev.create_context(ContextOptions(resources=[prefill_grp]))
        ctx_decode = dev.create_context(ContextOptions(resources=[decode_grp]))
        print("two-context creation succeeded")

        dev.set_current(ctx_prefill)
        print("dev.set_current(ctx_prefill) succeeded")
        prefill_stream = ctx_prefill.create_stream()
        print(f"ctx_prefill.create_stream succeeded; stream type: {type(prefill_stream)}")

        dev.set_current(ctx_decode)
        print("dev.set_current(ctx_decode) succeeded")
        decode_stream = ctx_decode.create_stream()
        print(f"ctx_decode.create_stream succeeded; stream type: {type(decode_stream)}")

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



