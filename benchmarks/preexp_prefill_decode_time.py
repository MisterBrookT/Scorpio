from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import argparse
import asyncio
import base64
import io
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple
from vllm.utils import FlexibleArgumentParser
import itertools

# Global request ID counter
request_id_counter = 0

def create_test_prompts(input_lens, output_lens, model_path):
    prompts = []
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
        prompts.append((candidate_prompt, SamplingParams(temperature=0, top_p=1.0, ignore_eos= True, min_tokens = output_len, max_tokens=output_len, tpot=1000000)))

    return prompts

def sample_prefill_time(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    global request_id_counter
    start_time = time.time()
    for (prompt, sampling_params) in test_prompts:
        engine.add_request(str(request_id_counter), prompt, sampling_params)
        request_id_counter += 1

    while engine.has_unfinished_requests():
        engine.step()
    round_finish_time = time.time() - start_time
    return round_finish_time

def sample_decode_time(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]],
                     special_token_idxs: List[int] = None):
    global request_id_counter
    step_time_list = []
    start_time = time.time()
    for (prompt, sampling_params) in test_prompts:
        engine.add_request(str(request_id_counter), prompt, sampling_params)
        request_id_counter += 1

    step = 0
    if special_token_idxs is not None:
        while engine.has_unfinished_requests():
            step += 1
            if step in special_token_idxs:
                step_start_time = time.time()
                engine.step()
                step_finish_time = time.time()
                step_time_list.append({"token_idx": step, "time": step_finish_time - step_start_time})
            else:
                engine.step()
    else:
        while engine.has_unfinished_requests():
            step += 1
            step_start_time = time.time()
            engine.step()
            step_finish_time = time.time()
            step_time_list.append({"token_idx": step, "time": step_finish_time - step_start_time})
    finish_time = time.time() - start_time
    return step_time_list, finish_time

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)

# prefill
def benchmark_prefill(engine, input_len_list, num_reqs, args):
    prefill_result = []
    for input_len, num_req in itertools.product(input_len_list, num_reqs):
        input_lens = [input_len for _ in range(num_req)]
        output_lens = [1 for _ in range(num_req)]
        test_prompts = create_test_prompts(input_lens=input_lens, output_lens=output_lens, model_path=args.model)
        prefill_result.append({"input_len": input_len, "num_req": num_req, "time": sample_prefill_time(engine, test_prompts)})
    return prefill_result

# benchmark decode with different output length, same batch size
def benchmark_decode_by_output_len(engine, num_reqs, args, input_len, output_len):
    decode_result = []
    for num_req in num_reqs:
        input_lens = [input_len for _ in range(num_req)]
        output_lens = [output_len for _ in range(num_req)]
        test_prompts = create_test_prompts(input_lens=input_lens, output_lens=output_lens, model_path=args.model)
        step_time_list, finish_time = sample_decode_time(engine, test_prompts)
        decode_result.append({"num_req": num_req, "duration": finish_time, "step_time_list": step_time_list})
    return decode_result

# benchmark decode with different batch size, same output length
def benchmark_decode_by_batch_size(engine, num_reqs, args, input_len, output_len, special_token_idxs = None):
    decode_result = []
    for num_req in num_reqs:
        input_lens = [input_len for _ in range(num_req)]
        output_lens = [output_len for _ in range(num_req)]
        test_prompts = create_test_prompts(input_lens=input_lens, output_lens=output_lens, model_path=args.model)
        step_time_list, finish_time = sample_decode_time(engine, test_prompts, special_token_idxs)
        decode_result.append({"num_req": num_req, "duration": finish_time, "step_time_list": step_time_list})
    return decode_result

def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    import itertools
    # Reset global counter at the start of main
    global request_id_counter
    request_id_counter = 0
    
    engine = initialize_engine(args)
    
    # prefill
    # num_reqs = [1, 4, 8, 16, 32]
    # input_len_list = [256, 512]
    # prefill_result = benchmark_prefill(engine, input_len_list, num_reqs, args)

    # decode_same_batch_different_output_len
    # num_reqs = [1,2,4,8,16]
    # decode_result_different_output_len = benchmark_decode_by_output_len(engine, num_reqs, args, input_len=1, output_len=2048)

    # decode_different_batch_same_output_len
    num_reqs = [1,2,4,8,16, 32, 64]
    decode_result_different_batch_size = benchmark_decode_by_batch_size(engine, num_reqs, args, input_len=256, output_len=32)

    # Save the result to a JSON file
    def save_result_to_json(data, filename, output_dir="benchmarks/result/pre_exp"):
        """Save benchmark results to a JSON file.
        
        Args:
            data: The benchmark data to save
            filename: Name of the output file
            output_dir: Directory to save the file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Results saved to {output_path}")
        return output_path
    
    # Save all benchmark results
    # save_result_to_json(prefill_result, f"{os.path.basename(args.model)}_prefill_result.json")
    # save_result_to_json(decode_result_different_output_len, f"{os.path.basename(args.model)}_decode_result_different_output_len.json")
    save_result_to_json(decode_result_different_batch_size, f"{os.path.basename(args.model)}_decode_result_different_batch_size.json")

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    main(args)
