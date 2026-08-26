system FMHA_System_v3 {

    // ============================================================
    // 1. Design Space (The "What" - Architectural Decisions)
    // ============================================================
    design_space {
        softmax_scheme: ["online", "naive"]
        math_approximation: ["exp2", "exp"]
        mask_fusion: ["fused_qk", "post_qk", "no_mask"]
        accum_dtype: ["f32", "f16"]
    }

    // ============================================================
    // 2. Tuning Space (The "How" - Tactical Parameters)
    // ============================================================
    tuning_space {
        tile_m: [32, 64, 128, 256]
        tile_n: [32, 64, 128, 256]
        tile_d: [32, 64, 128]
        occupancy: [1, 2, 4] // default 2
    }

    // ============================================================
    // 3. Model & Knowledge (The Semantic Layer)
    // ============================================================
    model {
        type KernelConfig matches { tile_m: int, tile_n: int, occupancy: int }
        state current_design: "online_fused"
    }

    knowledge {
        // Fundamental Correctness Invariant
        invariant Correctness {
            assert: "The output of FMHA must match the reference PyTorch implementation (within tolerance)."
            verification: "Run correctness test suite on target device."
        }
        
        // Semantic Invariants
        invariant NumericalStability {
            assert: "Online Softmax must subtract local maximum to prevent overflow in exp."
        }

        // S = Q @ K.T 는 절대 global memory에 저장되지 않음
        invariant MemoryEfficiency {
            assert: "Attention matrix (S = QK^T) must NOT be materialized in Global Memory."
        }

        // Abductive Rules (Knowledge Transfer)
        rule "Hide Latency" {
            when: "is_memory_bound"
            apply: "Increase occupancy target and use asynchronous loads (cp.async)."
        }

        rule "Precision Trade-off" {
            when: "is_compute_bound"
            apply: "Use exp2 and fast math approximations to increase TFLOPS."
        }

        fact is_memory_bound(t: TraceItem.Performance) {
            return t.speedup < 0.5 && t.tflops < 50.0
        }
    }

    // ============================================================
    // 4. Trace Schema (Polymorphic Observations)
    // ============================================================
    trace_schema {
        variant TraceItem {
            case Performance {
                step_name: string
                tflops: float
                speedup: float
            }
            case Correctness {
                step_name: string
                passed: boolean
                max_error: float
            }
        }
    }

    // ============================================================
    // 5. Loops (Execution Strategy)
    // ============================================================

    // Agent Loop: Strategic reasoning over the design space
    agent_loop FMHA_Architect {
        step "Identify Design Bottleneck" {
            llm.query { 
                prompt: "Analyze the current design {{current_design}} against MemoryEfficiency invariant."
                output_var: "bottleneck_analysis"
            }
        }

        step "Select Fusion Strategy" {
            llm.query {
                prompt: "Given {{bottleneck_analysis}}, choose best mask_fusion from design_space."
                output_var: "chosen_fusion"
            }
        }

        step "Generate Correct-by-Construction Kernel" {
            tool.write { 
                path: "fmha_generated.py" 
                content: "/* Generated based on {{chosen_fusion}} */" 
            }
        }
    }

    // Engineering Loop: Tactical sweep over the tuning space
    engineering_loop FMHA_Tuner {
        parameter Tile_M: [64, 128]
        parameter Tile_N: [64, 128]
        
        measure {
            cmd: "python fmha_generated.py --tile_m {{Tile_M}} --tile_n {{Tile_N}}"
            metric: "tflops"
            objective: "maximize"
        }
    }

    // ============================================================
    // 6. Build (Final Deliverables)
    // ============================================================
    build {
        artifact "Final_FMHA_Report.md" {
            generator: "python generate_fmha_report.py"
        }
    }
}
