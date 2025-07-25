from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random
from vllm.transformers_utils.tokenizer import get_tokenizer

model_path = "/root/autodl-pub/models/gemma-2-27b"
# model_path = "/root/autodl-pub/models/opt-13b"
# Sample prompts.
def create_prompts(input_lens, output_lens, ttft, tpot):
    prompts = []
    sampling_params = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    for input_len, output_len in zip(input_lens, output_lens):

        candidate_ids = [
                    random.randint(0, vocab_size - 1)
                    for _ in range(input_len)
                ]
        for _ in range(5):  # Max attempts to correct
            candidate_prompt = tokenizer.decode(candidate_ids)
            tokenized_len = len(tokenizer.encode(candidate_prompt))

            if tokenized_len == input_len:
                break

            # Adjust length based on difference
            diff = input_len - tokenized_len
            if diff > 0:
                candidate_ids.extend([
                    random.randint(100, vocab_size - 100)
                    for _ in range(diff)
                ])
            else:
                candidate_ids = candidate_ids[:diff]
        prompts.append(candidate_prompt)
        sampling_params.append(SamplingParams(temperature=0, top_p=1.0, ignore_eos= True, min_tokens = output_len, max_tokens=output_len, pred_output_len=output_len, ttft= ttft, tpot = tpot))
        # sampling_params.append(SamplingParams(temperature=0, top_p=1.0, ignore_eos= True, max_tokens=output_len))
    return prompts, sampling_params
prompts, sampling_params = create_prompts([256 for i in range(256)], [128 for i in range(256)], 5000, 50)
prompts, sampling_params = create_prompts([128 for i in range(256)], [256 for i in range(256)], 10000, 30)
# Create an LLM.
llm = LLM(model=model_path, max_model_len = 2048, block_size = 16, gpu_memory_utilization= 0.95,
          load_format = "dummy", enforce_eager=True,  scheduling_policy = "fcfs",tensor_parallel_size=4)


tokenizer = get_tokenizer(model_path)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    output_len = len(
    tokenizer(generated_text,
                add_special_tokens=False).input_ids)
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")