system FMHA_System_v2 {

    // 1. Model & Knowledge (Shared)
    model {
        type Config matches { tile_size: int, stages: int }
        state current_best: Config
    }

    // ============================================================
    // 2. Trace Schema (Unified with MatMul)
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
                component: string
            }
            case Analysis {
                bottleneck: "Memory" | "Compute" | "Latency"
            }
        }
    }

    // ============================================================
    // 3. Knowledge
    // ============================================================
    knowledge {
        fact is_converged(t: TraceItem.Performance) {
            return t.tflops > 100.0
        }
        
        fact is_verified(t: TraceItem.Correctness) {
            return t.passed == true
        }
    }

    // 2. Agent Loop: The "Architect"
    agent_loop Kernel_Architect {
        
        step "Generate Invariant Test" {
            tool.write { path: "fmha_step1_sim.py" }
            tool.run { cmd: "python fmha_step1_sim.py" }
            emit TraceItem.Correctness
        }

        step "Generate CUDA Kernel" {
            llm.query { 
                prompt: "Write a CuTile kernel for FMHA..." 
                output_var: "kernel_code"
            }
            tool.write { path: "fmha_kernel.py", content: "{{kernel_code}}" }
        }

        step "Review Optimization Results" {
            if "is_converged(trace.latest)" {
                emit "Optimization Complete"
            }
        }
    }

    // 3. Engineering Loop: The "Tuner"
    engineering_loop Kernel_Tuner {
        
        parameter Tile_M: [64, 128]
        parameter Tile_N: [32, 64, 128]
        
        measure {
            cmd: "python fmha_kernel.py --tile_m {{Tile_M}} --tile_n {{Tile_N}}"
            metric: "tflops"
            objective: "maximize"
        }
    }

    build {
        artifact "Final_FMHA_Report.md" {
            generator: "python generate_fmha_report.py"
            input: trace.all
        }
    }
}
