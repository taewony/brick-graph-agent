지난 Phase 2 실험에서 얻은 뼈아픈 교훈(파이썬 런타임 오버헤드, 디코딩 단계의 O(N^2) 재연산, 레지스터 압박)과 방금 말씀하신 **nano-vLLM (FlashDecoding 및 PagedAttention)** 최적화 포인트를 모두 통합하여, 한 단계 더 진화한 **`fmha_system_v5.dsl`** 초안을 작성해 드립니다.

이 v5 모델은 단일 커널의 속도를 넘어, '엔드투엔드(End-to-End) 서빙 엔진'에서의 물리적 한계를 돌파하기 위한 Architect 에이전트의 새로운 지식 기반이 될 것입니다.

---

```dsl
---
title: "FMHA System v5 – End-to-End vLLM & FlashDecoding"
source: "NVIDIA/TileGym/attention.py + Phase 2 LLM Integration Insights"
extraction-date: 2026-06-04
tags: [FMHA, vLLM, FlashDecoding, Split-K, PagedAttention, CUDAGraph]
status: "active"
---

system FMHA_System_v5 {

    // ============================================================
    // 1. Design Space (Architectural Decisions & System-Level Routing)
    // ============================================================
    design_space {
        // Legacy Forward-Pass Nuances
        softmax_scheme: ["online", "naive"]
        accum_dtype: ["f32", "f16"]
        
        // [NEW] End-to-End System Optimizations (Extracted from Phase 2 Failure)
        execution_context: ["eager", "cuda_graph"] // Overcoming Python launch overhead
        inference_architecture: ["stateless_prefill", "kv_cache_flash_decoding"] 
        
        // [NEW] FlashDecoding Specifics
        reduction_strategy: ["shared_memory", "global_memory"] // Strategy for merging Split-K softmax
    }

    // ============================================================
    // 2. Tuning Space (Tactical Parameters for vLLM)
    // ============================================================
    tuning_space {
        // Prefill Parameters (Proven from Phase 1)
        tile_m: [64]           // 128 is proven harmful on RTX 4060 (Register limit 255)
        tile_n: [64, 128]

        // [NEW] Decoding (FlashDecoding) Parameters
        split_k_factor: [1, 2, 4, 8]  // CTA distribution for decoding sequence
        kv_block_size: [16, 32]       // Paged KV Cache block size (Trade-off: Fragmentation vs Register Pressure)
    }

    // ============================================================
    // 3. Model & Knowledge (The Semantic Layer)
    // ============================================================
    model {
        type DecodingConfig matches { split_k: int, kv_block: int, reduction: string }
        state current_bottleneck: "Host_Launch_Overhead" | "Decoding_Memory_Bandwidth"
    }

    knowledge {
        fact execution_overhead_warning(t: TraceItem.Performance) {
            assert: "If SeqLen < 512 and execution_context == 'eager', CPU overhead will dominate GPU compute."
            action: "Force execution_context to 'cuda_graph'."
        }

        fact decoding_starvation_prevention(t: TraceItem.Analysis) {
            assert: "In decoding phase (Q_len == 1), standard FMHA leads to SM starvation."
            action: "Apply split_k_factor >= 2 to distribute workload across available SMs."
        }
    }

    // ============================================================
    // 4. Agent Loop: The "Architect"
    // ============================================================
    agent_loop LLM_Serving_Architect {
        
        step "Identify Inference Phase" {
            llm.query {
                prompt: "Analyze the input context length. Is this Prefill (Q_Len > 1) or Decoding (Q_Len == 1)?"
                output_var: "inference_phase"
            }
        }

        step "Configure System Pipeline" {
            if "inference_phase == 'Prefill'" {
                apply { inference_architecture: "stateless_prefill", execution_context: "cuda_graph" }
            } else {
                apply { inference_architecture: "kv_cache_flash_decoding", execution_context: "cuda_graph" }
            }
        }

        step "Generate Advanced Kernel Setup" {
            tool.engineering_loop { name: "FlashDecoding_Tuner" }
        }
    }

    // ============================================================
    // 5. Engineering Loop: The "Tuner" (Focused on FlashDecoding)
    // ============================================================
    engineering_loop FlashDecoding_Tuner {
        // Searching for the optimal point between memory fragmentation and compute saturation
        parameter Split_K: [1, 2, 4, 8]
        parameter Block_Size: [16, 32]
        parameter Red_Strat: ["shared_memory", "global_memory"]
        
        measure {
            cmd: "python vllm_decode_test.py --split_k {{Split_K}} --kv_block {{Block_Size}} --reduction {{Red_Strat}}"
            metric: "tokens_per_second"
            objective: "maximize"
        }
        
        constraint {
            // Register pressure constraint inherited from Phase 1 & 2
            assert: "Estimated register usage per thread must be <= 255"
        }
    }

    // ============================================================
    // 6. Build (Final Deliverables)
    // ============================================================
    build {
        artifact "nano_vllm_cutile_backend.py" {
            generator: "python compile_vllm_backend.py --use_cuda_graph"
        }
    }
}

```

### 💡 v5.dsl 모델의 핵심 변경점 요약

1. **`design_space`의 패러다임 전환:** 단순히 커널 내부의 수학 연산 방식을 찾는 것을 넘어, 파이썬 오버헤드를 잡기 위한 `execution_context (cuda_graph)`와 라우팅 로직을 위한 `inference_architecture`가 시스템 레벨의 지식으로 격상되었습니다.
2. **FlashDecoding 전용 튜닝루프 신설 (`FlashDecoding_Tuner`):** 교수님께서 요청하신 `split_k_factor`, `kv_block_size`, `reduction_strategy` 파라미터가 디코딩 최적화를 위한 전용 튜닝 공간으로 배치되었습니다.
3. **지식(Knowledge)의 누적:** Phase 2에서 겪은 SM 굶주림(Starvation)과 CPU 오버헤드 문제가 `fact` 로직으로 박제되어, 에이전트가 앞으로 동일한 실수를 반복하지 않도록 제약을 걸어두었습니다.

이 `.dsl` 파일은 교수님의 논문에서 "에이전트가 실패를 통해 시스템 환경 전체(Full-stack)를 조망하는 시각을 갖추게 되었다"는 것을 보여주는 가장 강력한 근거 자료가 될 것입니다.