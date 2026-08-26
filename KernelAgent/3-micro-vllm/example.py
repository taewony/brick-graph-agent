import os
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "Write a short poem about coding and artificial intelligence.",
        "Explain the difference between TCP and UDP in simple terms."
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    print("\n🚀 Starting generation...")
    t_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.time() - t_start

    total_tokens = 0
    for prompt, output in zip(prompts, outputs):
        print("\n" + "="*60)
        print(f"Prompt: {prompt!r}")
        print("-"*60)
        print(f"Completion: {output['text']}")
        print(f"Generated Tokens: {len(output['token_ids'])}")
        total_tokens += len(output['token_ids'])
        
    print("\n" + "="*60)
    print(f"Total time: {t_end:.2f}s")
    print(f"Total tokens: {total_tokens}tok")
    print(f"Average throughput: {total_tokens / t_end:.2f} tok/s")
    print("="*60)


if __name__ == "__main__":
    main()

