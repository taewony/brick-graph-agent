---
title: "FMHA System v3 – Compound Engineering Ready"
source: "NVIDIA/TileGym/attention.py + Gemini CLI auto-insight"
extraction-date: 2026-02-13
tags: [FMHA, cuTile, compound, auto-extracted]
status: "active"
---

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
        }

        // S = Q @ K.T 는 절대 global memory에 저장되지 않음
        invariant MemoryEfficiency {
            assert: "Attention matrix (S = QK^T) must NOT be materialized in Global Memory."
        }
        
        // Semantic Invariants
        invariant NumericalStability {
            assert: "Online Softmax must subtract local maximum to prevent overflow in exp."
        }

        // Abductive Rules (Knowledge Transfer)
        rule "Hide Latency" {
            when: "is_memory_bound"
            apply: "Increase occupancy target and use asynchronous loads (cp.async)."
        }

        // ----- Discovered Facts (from Report/Experiments) -----
        fact optimal_tile_size_for_rtx5070 {
            description: "On RTX 5070, tile_m = 64, tile_n = 64 yield best performance."
            confidence: 0.95
            source: "Final_FMHA_Report.md Step 4"
        }
        
        fact naive_kernel_memory_bound {
            description: "Naive FMHA kernel is severely bound by global memory bandwidth (speedup < 0.05)."
            confidence: 0.99
            source: "Final_FMHA_Report.md Step 2"
        }

        // ----- Generalized Design Rules (Abductive) -----
        rule "Memory Bottleneck → Fuse QK and PV" {
            when: "is_memory_bound && operation == 'attention'"
            apply: "select mask_fusion = 'fused_qk'"
            evidence: "Step 2→3 speedup jump 0.02x → 0.60x"
            source: "Final_FMHA_Report.md Step 3"
        }

        rule "RTX50 Series Tile Size Heuristic" {
            when: "device_family == 'RTX50' && architecture == 'Blackwell'"
            recommend: "tile_m = 64, tile_n = 64 as initial guess"
            confidence: 0.9
            source: "Final_FMHA_Report.md Step 4"
        }

        // ----- Performance Classification -----
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
            case DesignCheck {
                invariant_name: string
                is_satisfied: boolean
            }
            case CodePattern {
                pattern_name: string      // 예: "exp2_approximation"
                location: string          // 파일명:라인
                confidence: float
            }
            case Insight {
                source: string
                category: string   // "bottleneck", "optimal", "rule", "negative"
                content: string
                confidence: float
                extracted_by: string
                applied: boolean
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
        
        // --- Engineering Loop 호출 ---
        step "Run Tuner" {
            tool.engineering_loop { name: "FMHA_Tuner" }
        }

        // --- 보고서 생성 ---
        step "Generate Report" {
            tool.run { 
                cmd: "python generate_fmha_report.py --template fmha_report.j2 --output Final_FMHA_Report.md"
            }
        }

        // --- 🚀 COMPOUND ENGINEERING CORE 🚀 ---
        step "Extract Insights from Report" {
            llm.query {
                input: file("Final_FMHA_Report.md")
                prompt: """
                    Extract structured insights from this FMHA engineering report.

                    Focus on:
                    - What bottleneck was eliminated and how?
                    - What numerical values were found optimal?
                    - What design decisions contributed most to speedup?
                    - What parameters were ineffective?
                    
                    Return JSON array with fields: category, content, confidence, source_line.
                    """
                output_var: "new_insights"
            }
        }

        step "Integrate Insights into DSL" {
            tool.dsl_integrate {
                dsl: "fmha_system_v4.dsl"
                insights: "{{new_insights}}"
                // 자동으로 knowledge, constraints 업데이트
            }
        }

        step "Update External Knowledge Base" {
            tool.write {
                path: "compound_knowledge_base/fmha_patterns.md"
                content: "## Auto-extracted Patterns ({{date}})\n{{new_insights}}"
                mode: "append"
            }
        }

        step "Verify Autonomous Detection" {
            llm.query {
                prompt: """
                    Given the newly added insights in the DSL,
                    if we encounter a similar memory-bound attention kernel in the future,
                    can the system automatically recommend fusion? 
                    Identify any gaps in the current rule set.
                    """
                output_var: "autonomy_gap_analysis"
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
        artifact "FMHA_cuTile_DesignSpace.dsl" {
            type: "KnowledgeExport"
            content: design_space + tuning_space + knowledge
        }
        artifact "Final_FMHA_Report.md" {
            generator: "python generate_fmha_report.py"
        }
        
        artifact "compound_insights_log.json" {
            generator: "collect_insights"
        }
    }
}
