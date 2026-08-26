import os
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Initialize LLM in eager mode for native Windows execution
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=100)

    # 1. User 1 submits a prompt immediately
    llm.add_request("Explain the theory of relativity in one sentence.", sampling_params)
    print("📥 User 1 request added.")

    step_count = 0

    # 2. Run the generation iteratively
    finished_outputs = {}
    while not llm.is_finished():
        step_count += 1

        # Simulate User 2 submitting a request dynamically at Step 5
        if step_count == 5:
            llm.add_request("What is the capital of France?", sampling_params)
            print("\n📥 User 2 request added dynamically while User 1 is decoding!")

        # Execute a single engine step
        outputs, num_tokens = llm.step()
        for seq_id, token_ids in outputs:
            finished_outputs[seq_id] = token_ids

        # Print active generations at each step
        active_ids = [seq.seq_id for seq in llm.scheduler.running]
        print(f"Step {step_count}: Active User IDs={active_ids}, Tokens processed={num_tokens}")

    # 3. Print final outputs
    print("\n" + "=" * 60)
    for seq_id in sorted(finished_outputs.keys()):
        token_ids = finished_outputs[seq_id]
        text = tokenizer.decode(token_ids)
        print(f"User {seq_id} Completion: {text.strip()}\n" + "-" * 60)



if __name__ == "__main__":
    main()
