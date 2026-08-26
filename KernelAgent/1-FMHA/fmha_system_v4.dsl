---
title: "FMHA System v4 – Forward Pass Deep Dive"
source: "NVIDIA/TileGym/attention.py"
extraction-date: 2026-02-13
tags: [FMHA, GQA, TMA, CompoundEngineering]
status: "active"
---

system FMHA_System_v4 {

    // ============================================================
    // 1. Design Space (Architectural Decisions)
    // ============================================================
    design_space {
        // Core Logic
        softmax_scheme: ["online", "naive"]
        math_approximation: ["exp2", "exp"]
        mask_fusion: ["fused_qk", "post_qk", "no_mask"]
        accum_dtype: ["f32", "f16"]
        causal: [true, false]

        // New Forward-Pass Nuances (Extracted from v4)
        attention_variant: ["MHA", "GQA", "MQA"]
        kernel_mode: ["inference", "training"] // training saves LSE
        memory_robustness: ["assume_aligned", "dynamic_k_masking"] // EVEN_K flag
        load_latency_strategy: ["default", "tma_optimized"] // latency hints
    }

    // ============================================================
    // 2. Tuning Space (Tactical Parameters)
    // ============================================================
    tuning_space {
        tile_m: [32, 64, 128, 256]
        tile_n: [32, 64, 128, 256]
        tile_d: [32, 64, 128]
        occupancy: [1, 2, 4] // default 2
        
        // Strategy-dependent parameters
        k_load_latency: [1, 2, 3]
        v_load_latency: [1, 2, 3, 4, 5]
    }

    // ============================================================
    // 3. Model & Knowledge (The Semantic Layer)
    // ============================================================
    model {
        type KernelConfig matches { tile_m: int, tile_n: int, occupancy: int }
        state current_design: "online_fused_gqa"
    }

    knowledge {
        // --- Invariants ---
        invariant Correctness {
            assert: "The output of FMHA must match the reference PyTorch implementation."
        }
        
        invariant GQA_Safety {
            assert: "num_heads % num_head_kv == 0"
            source: "attention.py:L716"
        }

        // --- Verified Facts (RTX 5070 Final Peak) ---
        fact optimal_rtx5070_config {
            description: "On RTX 5070, DSL-generated kernel achieves 135.03 TFLOPS, outperforming PyTorch Native (121.38 TFLOPS)."
            causal_tflops: 135.03
            torch_baseline_tflops: 121.38
            speedup_vs_torch: 1.11
            correctness: "Verified (100% match with 1e-5 tolerance)"
            confidence: 1.0
            source: "last_engineering_trace.json"
        }

        fact causal_loop_speedup {
            description: "Causal loop optimization (m_end pruning) provides ~1.8x effective throughput increase."
            confidence: 1.0
            source: "last_engineering_trace.json comparison"
        }

        fact negative_pattern_tile_m_128 {
            description: "Tile_M=128 causes severe performance degradation on RTX 5070 (throughput cut by >50%)."
            confidence: 1.0
            source: "last_engineering_trace.json"
        }

        // --- Abductive Rules (Compound Logic) ---
        rule "Blackwell TMA Pipelining" {
            when: "device == 'NVIDIA GeForce RTX 5070'"
            recommend: "k_load_latency = 2"
            recommend: "v_load_latency = 5 if causal else 4"
            evidence: "Optimal configs favor deeper V-load pipelining in causal mode."
        }

        rule "Tile Size Heuristic" {
            when: "device == 'NVIDIA GeForce RTX 5070'"
            recommend: "tile_m = 64"
            reason: "Experimental data shows Tile_M=128 results in >50% throughput loss (likely due to occupancy/register pressure)."
            confidence: 0.99
        }

        rule "GQA Mapping" {
            when: "num_heads > num_head_kv"
            apply: "select attention_variant = 'GQA'"
            logic: "off_kv_h = head_idx // (num_heads // num_head_kv)"
            source: "attention.py:L64"
        }

        rule "Non-Aligned K Handling" {
            when: "k_len % tile_n != 0"
            apply: "select memory_robustness = 'dynamic_k_masking'"
            action: "Add (offs_n < k_seqlen) to mask logic"
            source: "attention.py:L120"
        }

        rule "TMA Optimization" {
            when: "load_latency_strategy == 'tma_optimized'"
            apply: "use ct.load(..., latency=k_load_latency) for K"
            apply: "use ct.load(..., latency=v_load_latency) for V"
        }
    }

    // ============================================================
    // 4. Trace Schema (Polymorphic Observations)
    // ============================================================
    trace_schema {
        variant TraceItem {
            case Performance { step_name: string, tflops: float, speedup: float }
            case Correctness { step_name: string, passed: boolean, max_error: float }
            case CodePattern { pattern_name: string, location: string }
            case Insight { category: string, content: string, source: string }
        }
    }

    // ============================================================
    // 5. Loops (Execution Strategy)
    // ============================================================

    agent_loop FMHA_Forward_Refiner {
        step "Analyze Source Patterns" {
            llm.query {
                prompt: "Identify hardware-specific hints (TMA, latency) in the provided source code."
                output_var: "hw_hints"
            }
        }

        step "Select Strategy" {
            llm.query {
                prompt: "Given {{hw_hints}}, choose best load_latency_strategy."
                output_var: "chosen_strategy"
            }
        }

        step "Generate and Tune" {
            tool.engineering_loop { name: "FMHA_Forward_Tuner" }
        }
        
        step "Finalize Forward Design" {
            tool.write { path: "fmha_final_v4.py", content: "..." }
        }
    }

    engineering_loop FMHA_Forward_Tuner {
        parameter Tile_M: [64]          // 128 is proven harmful on this device
        parameter Tile_N: [64, 128]     // 128 yields ~55 TFLOPS - useful for memory-bound tradeoffs
        parameter K_Lat: [2, 3]         // Both values are viable for Blackwell TMA
        parameter V_Lat: [4, 5]         // Both values are viable for Blackwell TMA
        parameter Causal: [0, 1]        // Sweep both non-causal and causal
        
        measure {
            cmd: "python fmha_v4_test.py --tile_m {{Tile_M}} --tile_n {{Tile_N}} --klat {{K_Lat}} --vlat {{V_Lat}} --causal {{Causal}}"
            metric: "tflops"
            objective: "maximize"
        }
    }

    // ============================================================
    // 6. Build (Final Deliverables)
    // ============================================================
    build {
        artifact "fmha_system_v4.dsl"
        artifact "insights_log.jsonl"
    }
}
