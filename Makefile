# brick.agent — common commands (brick-agent-plan Phase 8.5)
# Usage: make test | make seed | make ask-sql Q="..." | make ask-okf Q="..."
PY ?= .wenv/Scripts/python.exe

.PHONY: test seed ask-sql ask-okf ask ingest lint mentor-status browse repair validate

## Run the full test suite (65+ tests)
test:
	$(PY) -m pytest tests/ -q

## Seed data/store_front.db from .okf/02_store_front
seed:
	$(PY) -m src.agents.brick_agent db seed

## SQL Q&A: make ask-sql Q="How many stable tools are in the store front?"
ask-sql:
	$(PY) -m src.agents.brick_agent ask sql "$(Q)"

## OKF Q&A: make ask-okf Q="How does prefill_phase relate to decode_phase?"
ask-okf:
	$(PY) -m src.agents.brick_agent ask okf "$(Q)"

## Router-classified ask: make ask Q="List the stable decisions"
ask:
	$(PY) -m src.agents.brick_agent ask "$(Q)"

## Ingest the nano-vLLM KB
ingest:
	$(PY) -m src.agents.brick_agent ingest --kb-id nano --kb-path .okf/01_nano_vllm

## Lint the nano-vLLM KB (expect: valid=True n_errors=0)
lint:
	$(PY) -m src.agents.brick_agent lint --kb-id nano --kb-path .okf/01_nano_vllm

## Show pipelines / guardrails / stored runs / LLM usage
mentor-status:
	$(PY) -m src.agents.brick_agent mentor status

## Generate + open the OKF KB web dashboard (docs/<kb>/index.html)
##   make browse KB=01_nano_vllm | make browse KB=".okf/01_nano_vllm"
browse:
	$(PY) -m src.agents.brick_agent browse --kb "$(KB)"

## Re-run the KB repair tool (idempotent; expects lint-clean)
repair:
	$(PY) -m src.tools.okf_repair

## OKF v0.2 conformance validation
validate:
	$(PY) src/okf/validator.py .okf/01_nano_vllm
