r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import datasets
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
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput, RequestFuncOutput, SchedulingMetric, get_scheduling_metric)
from datasets import load_dataset
from PIL.Image import Image
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from benchmarks.backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser
import os
import csv
import pandas as pd

MILLISECONDS_TO_SECONDS_CONVERSION = 1000

# Define a typed dictionary for request data structure
class RequestData(TypedDict, total=False):
    prompt: str
    prompt_len: int
    output_len: int
    ttft: int
    tpot: int
    timestamp: Optional[datetime]


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    completion_summary: List
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    mean_top: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]
    tpots: List
    
    trace_slo_adherence: List[Tuple[datetime, bool]]



def sample_from_local(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[RequestData]:
    # Set fixed seed for reproducibility
    random.seed(42)

    if dataset_path.endswith('.jsonl'):
        # Process JSONL file
        # First read all lines into a list
        all_lines = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_lines.append(line)
        print(f"len(all_lines): {len(all_lines)}")
        filtered_dataset: List[RequestData] = []


        # Shuffle the lines if the first line has no timestamp
        # Check if the first line is a valid JSON and has a timestamp field
        first_line_data = json.loads(all_lines[0])
        if 'timestamp' not in first_line_data:
            random.shuffle(all_lines)

        for line in all_lines:
            if len(filtered_dataset) == num_requests:
                break
            
            data = json.loads(line)
            prompt = data['prompt']
            prompt_len = len(tokenizer(prompt).input_ids)
            
            
            
            # If the JSON already has generated text, use it to calculate output length
            # Otherwise use a default or estimate
            if fixed_output_len is not None:
                output_len = fixed_output_len
            else:
                assert 'generated' in data, "Generated text not found in JSONL data"
                generated_text = data['generated']
                output_len = len(tokenizer(generated_text).input_ids)

            if fixed_output_len is  None:
                if prompt_len < 4 or  output_len < 4:
                    # Prune too short sequences.
                    continue
                if prompt_len > 1024 or  prompt_len + output_len > 2048:
                    # Prune too long sequences.
                    continue

            # Extract SLO values if they exist, otherwise use default values
            if 'ttft' in data and 'tpot' in data:
                ttft = data['ttft']
                tpot = data['tpot']
            else:
                # Random selection of one of three candidate SLO types
                slo_type = random.choice([0, 1, 2])
                if slo_type == 0:
                    ttft, tpot = 500, 50    # type 0
                elif slo_type == 1:
                    ttft, tpot = 3000, 30   # type 1
                else:
                    ttft, tpot = 15000, 50  # type 2
            
            # Extract timestamp if it exists
            timestamp = None
            if 'timestamp' in data:
                try:
                    timestamp = datetime.fromisoformat(data['timestamp'])
                except (ValueError, TypeError):
                    # If timestamp format is invalid, just use None
                    pass
            
            filtered_dataset.append(RequestData(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                ttft=ttft,
                tpot=tpot,
                timestamp=timestamp
            ))
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}. Only jsonl files are supported.")
    print(f"len(filtered_dataset): {len(filtered_dataset)}")
    return filtered_dataset


def sample_from_sharegpt_lmsys(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
    start: int = 0,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    if dataset_path == "sharegpt":
        with open("datasets/ShareGPT_V3_unfiltered_cleaned_split.json") as f:
            dataset = json.load(f)
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        dataset = dataset[start:start + int(num_requests * 1.2)] 
        ds = dataset
        # Only keep the first two turns of each conversation.
        dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]
        prompts = []
        for prompt, _ in dataset:
            # Format for Gemma with correct turn markers
            chat = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)
    elif dataset_path == "lmsys":
        dataset = datasets.load_dataset("lmsys/lmsys-chat-1m")['train']
        ds = dataset.select(range(start, start + int(num_requests * 1.2)))
        prompts = []
        for i, question in enumerate(ds):
            prompt = None
            for convsat in question['conversation']:
                if convsat['role'] == 'user':
                    prompt = convsat['content']
                    break
            if prompt is None:
                continue
            # Format for Gemma with correct turn markers
            chat = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)

    prompt_token_ids = tokenizer(prompts).input_ids
    tokenized_dataset = []
    for i in range(len(prompts)):
        output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    filtered_dataset: List[str] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2000000: #only filter too long prompt
            # Prune too long sequences.
            continue
        filtered_dataset.append(RequestData(prompt=prompt, prompt_len=prompt_len, output_len=output_len-prompt_len, ttft=100000, tpot=100000, generation_mode=True))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)

    return sampled_requests

def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[RequestData]:
  
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode([(i + j) % tokenizer.vocab_size for j in range(input_len)])

        input_requests.append(RequestData(prompt=prompt,
                               prompt_len=input_len,
                               output_len=output_len,
                               ttft=None,
                               tpot=None,
                               timestamp=None))
    return input_requests

def parse_list_arg(arg_value):
    """Parse a string representation of a list into a Python list."""
    if isinstance(arg_value, str):
        # Remove brackets and split by commas
        try:
            # Handle string format like "[128,512]" or "128,512"
            clean_str = arg_value.strip('[]')
            return [int(x.strip()) for x in clean_str.split(',')]
        except ValueError:
            raise ValueError(f"Could not parse list argument: {arg_value}")
    elif isinstance(arg_value, list):
        return arg_value
    else:
        return [arg_value]  # Single value as a list


async def get_request(
    input_requests: List[RequestData],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[RequestData, None]:
    """
    Asynchronously generates requests at a specified rate or based on timestamps.
    
    Args:
        input_requests: 
            A list of input requests, each represented as a dictionary containing request data.
        request_rate: 
            The rate at which requests are generated (requests/s).
            Ignored if timestamps are present in the input_requests.
        burstiness (optional): 
            The burstiness factor of the request generation. 
            Only takes effect when request_rate is not inf and timestamps are not used.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results 
            in more bursty requests, while a higher burstiness value 
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = list(input_requests)
    
    # Check if timestamps are included in the requests
    has_timestamps = input_requests[0].get("timestamp") is not None
    print(has_timestamps)
    if has_timestamps:
        print("Using timestamps from dataset for request timing")
        # Sort requests by timestamp
        input_requests.sort(key=lambda x: x["timestamp"])
        
        # Get the first timestamp as the base time
        base_time = input_requests[0]["timestamp"]
        first_request = True
        
        for i, request in enumerate(input_requests):
            if first_request:
                # No delay for the first request
                first_request = False
            else:
                # Calculate time difference from the base time in seconds
                time_diff = (request["timestamp"] - base_time).total_seconds()
                # Wait for the appropriate interval
                await asyncio.sleep(time_diff)
                # Update the base time for the next request
                base_time = request["timestamp"]
            
            yield request
    else:
        # Original behavior with Poisson or gamma distribution
        # Calculate scale parameter theta to maintain the desired request_rate.
        assert burstiness > 0, (
            f"A positive burstiness factor is expected, but given {burstiness}.")
        theta = 1.0 / (request_rate * burstiness)

        for request in input_requests:
            yield request

            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the gamma distribution.
            # If burstiness is 1, it follows exponential distribution.
            interval = np.random.gamma(shape=burstiness, scale=theta)
            # The next request will be sent after the interval.
            await asyncio.sleep(interval)

def calculate_metrics(
    input_requests: List[RequestData],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    gootput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    ttft_violate = 0
    tpot_violate = 0
    ttft_violation_list = []
    tpot_violation_list = []
    both_violate = 0
    both_violation_list = []
    good_completed = 0
    reject_list = {}
    rejected = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    tops: List[float] = []
    # here true means the request is a good completion
    trace_slo_adherence: List[Tuple[datetime, bool]] = []
    for i in range(len(outputs)):
        ttft_objective = input_requests[i]["ttft"]/MILLISECONDS_TO_SECONDS_CONVERSION
        tpot_objective = input_requests[i]["tpot"]/MILLISECONDS_TO_SECONDS_CONVERSION
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly

            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i]["prompt_len"]
            

            tpot = 0
            if output_len > 1:
                tpot = (outputs[i].latency - outputs[i].ttft) / (output_len -
                                                                 1)
                tpots.append(tpot)
                
            tpot_meet = (tpot <= tpot_objective)
            ttft_meet = (outputs[i].ttft <= ttft_objective)
            if tpot_meet and ttft_meet:
                good_completed += 1
            elif tpot_meet and not ttft_meet:
                ttft_violate += 1
                ttft_violation_list.append({"TTFT_obj":ttft_objective, "TTFT_actual":outputs[i].ttft})

            elif not tpot_meet and ttft_meet:
                tpot_violate += 1
                tpot_violation_list.append({"TPOT_obj":tpot_objective, "TPOT_actual":tpot})
            else:
                both_violate += 1
                both_violation_list.append({"TTFT_obj":ttft_objective, "TTFT_actual":outputs[i].ttft, "TPOT_obj":tpot_objective, "TPOT_actual":tpot})
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            tops.append(outputs[i].top)
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            
            trace_slo_adherence.append((input_requests[i]["timestamp"], tpot_meet and ttft_meet))
            completed += 1
        elif outputs[i].reject:
            rejected += 1
            if reject_list.get(f"TTFT_obj:{ttft_objective}, TPOT_obj:{tpot_objective}", None) is None:
                reject_list[f"TTFT_obj:{ttft_objective}, TPOT_obj:{tpot_objective}"] = 0
            reject_list[f"TTFT_obj:{ttft_objective}, TPOT_obj:{tpot_objective}"] += 1
            actual_output_lens.append(0)
            
            trace_slo_adherence.append((input_requests[i]["timestamp"], False))
        else:
            actual_output_lens.append(0)
            
            trace_slo_adherence.append((input_requests[i]["timestamp"], False))

    assert ttft_violate+tpot_violate+both_violate+good_completed == completed

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        completion_summary = [good_completed, ttft_violate, tpot_violate, both_violate, rejected, ttft_violation_list, tpot_violation_list, both_violation_list, reject_list],
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
        tpots= tpots,
        mean_top= np.mean(tops or 0) * 1000,
        trace_slo_adherence=trace_slo_adherence
    )


    return metrics, actual_output_lens


async def benchmark(
    args,
    backend: str,
    engine_api_url: str,  # Changed from api_url to engine_api_url
    engine_base_url: str,  # Changed from base_url to engine_base_url
    predictor_api_url: str,  # Changed from predictor_url to predictor_api_url
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[RequestData],
    logprobs: Optional[int],
    best_of: int,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    gootput_config_dict: Dict[str, float],
    max_concurrency: Optional[int],
):
                
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_request = input_requests[0]
    generation_mode = args.dataset_name == "generation"


    print(f"*****generation_mode: {generation_mode}*****")


    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_request["prompt"], 
        engine_api_url=engine_api_url,  # Update to use engine_api_url
        predictor_api_url=predictor_api_url,  # Update to use 
        generation_mode=generation_mode,
        prompt_len=test_request["prompt_len"],
        output_len=test_request["output_len"],
        ttft = test_request["ttft"],
        tpot = test_request["tpot"],
        logprobs=logprobs,
        best_of=best_of,
        ignore_eos=ignore_eos,
    )
    # print(test_input)
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success and not test_output.reject:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    # Profile API URL handling
    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(model=model_id,
                                         prompt=test_request["prompt"],
                                         engine_api_url=engine_base_url + "/start_profile",  # Changed from api_url to engine_api_url
                                         prompt_len=test_request["prompt_len"],
                                         output_len=test_request["output_len"],
                                         ttft = test_request["ttft"],
                                         tpot = test_request["tpot"],
                                         logprobs=logprobs,
                                         best_of=best_of,
                                         ignore_eos=ignore_eos)
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")
    
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else None)

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness):
        request_func_input = RequestFuncInput(model=model_id,
                                              prompt=request["prompt"],
                                              engine_api_url=engine_api_url,  # Changed from api_url to engine_api_url
                                              predictor_api_url=predictor_api_url,  # Changed from predictor_url to predictor_api_url
                                              generation_mode=generation_mode,
                                              prompt_len=request["prompt_len"],
                                              output_len=request["output_len"],
                                              ttft = request["ttft"],
                                              tpot = request["tpot"],
                                              logprobs=logprobs,
                                              best_of=best_of,
                                              ignore_eos=ignore_eos)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input,
                                     pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if generation_mode:
        for i in range(len(outputs)):
            result_json = {"prompt": input_requests[i]["prompt"], "generated": outputs[i].generated_text}
            with open(f"gemma27b_lmsys_generation_results.jsonl", "a") as outfile:
                outfile.write(json.dumps(result_json) + "\n")
        return outputs
    
    # Change from api_url to engine_base_url for scheduler stats
    scheduling_metric: SchedulingMetric = get_scheduling_metric(api_url=f"{engine_base_url}/v1/scheduler/stats")

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_request["prompt"],
            engine_api_url=engine_base_url + "/stop_profile",  # Changed from api_url to engine_api_url
            prompt_len=test_request["prompt_len"],
            output_len=test_request["output_len"],
            ttft = test_request["ttft"],
            tpot = test_request["tpot"],
            logprobs=logprobs,
            best_of=best_of,
            ignore_eos=ignore_eos,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        gootput_config_dict=gootput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    if gootput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):",
                                        metrics.request_goodput))
    
    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "policy": args.scheduling_policy,
        "good_completion": metrics.completion_summary[0],
        "ttft_violation": metrics.completion_summary[1],
        "tpot_violation": metrics.completion_summary[2],
        "both_violation": metrics.completion_summary[3],
        "rejected": metrics.completion_summary[4],
        "ttft_violation_list": metrics.completion_summary[5],
        "tpot_violation_list": metrics.completion_summary[6],
        "both_violation_list": metrics.completion_summary[7],
        "reject_list": metrics.completion_summary[8],
        "scheduler_profile": scheduling_metric.scheduler_profile,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput,
        "mean_top": metrics.mean_top,
    }

    if not args.simple_result:
        result["total_input_tokens"] = metrics.total_input
        result["total_output_tokens"] = metrics.total_output
        result["total_token_throughput"] = metrics.total_token_throughput
        result["input_lens"] = [output.prompt_len for output in outputs]
        result["output_lens"] = actual_output_lens
        result["ttfts"] = [output.ttft for output in outputs]
        result["tpots"] = metrics.tpots
        result["itls"] = [output.itl for output in outputs]
    # print(f"tpots in benchmark: {metrics.tpots}")

    if args.trace_result:
        
        result["trace_slo_adherence"] = metrics.trace_slo_adherence


    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    gootput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        gootput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in gootput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
    return gootput_config_dict


def parse_goodput(slo_pairs):
    gootput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            gootput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds.") from err
    return gootput_config_dict


def datetime_to_str(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, list):

        return [datetime_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(datetime_to_str(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: datetime_to_str(v) for k, v in obj.items()}
    return obj

def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    # Use the new parameter structure for constructing URLs
    engine_api_url = f"http://{args.engine_host}:{args.engine_port}{args.engine_endpoint}"
    engine_base_url = f"http://{args.engine_host}:{args.engine_port}"
    
    if "gate" in args.scheduling_policy or "pred" in args.scheduling_policy: 
        predictor_api_url = f"http://{args.predictor_host}:{args.predictor_port}{args.predictor_endpoint}"
    else:
        predictor_api_url = None
    
    tokenizer = get_tokenizer(tokenizer_id,
                              tokenizer_mode=tokenizer_mode,
                              trust_remote_code=args.trust_remote_code) 

    if args.dataset_name == "local":
        input_requests = sample_from_local(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer= tokenizer,
            fixed_output_len=args.fixed_output_len,
        )
    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.input_len,
            output_len=args.output_len,
            num_prompts=args.num_prompts,
            tokenizer=tokenizer,
        )
    elif args.dataset_name == "generation":
        input_requests = sample_from_sharegpt_lmsys(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.output_len,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    gootput_config_dict = check_goodput_args(args)

    benchmark_result = asyncio.run( 
        benchmark(
            args = args,
            backend=backend,
            engine_api_url=engine_api_url,  # Use engine API URL here
            engine_base_url=engine_base_url,  # Use engine base URL here
            predictor_api_url=predictor_api_url,  # Use predictor API URL here
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            best_of=args.best_of,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            gootput_config_dict=gootput_config_dict,
            max_concurrency=args.max_concurrency,
        ))
    if args.dataset_name == "generation":
        print(f"generation requests finished.")
        return benchmark_result
    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["num_prompts"] = args.num_prompts
        result_json["slo_adherence_ratio"] = benchmark_result["good_completion"] / args.num_prompts
        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")
        # result_json["burstiness"] = args.burstiness
        # result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (f"-concurrency{args.max_concurrency}"
                               if args.max_concurrency is not None else "")
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
            
        if args.result_dir and not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        with open(file_name, "w", encoding='utf-8') as outfile:
            try:
                # Convert datetime objects to strings before saving
                result_json = datetime_to_str(result_json)
                json.dump(result_json, outfile)
                print(f"save file successfully to {file_name}")
            except Exception as e:
                print(f"Error saving file: {str(e)}")
                print(f"Error type: {type(e).__name__}")



if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    inference_engine_server = parser.add_argument_group("inference server ip and port")
    inference_engine_server.add_argument("--engine-host", type=str, default="localhost")
    inference_engine_server.add_argument("--engine-port", type=int, default=8000)
    inference_engine_server.add_argument(
        "--engine-endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )

    predictor_server = parser.add_argument_group("predictor server ip and port")
    predictor_server.add_argument("--predictor-host", type=str, default="localhost")
    predictor_server.add_argument("--predictor-port", type=int, default=8001)
    predictor_server.add_argument("--predictor-endpoint", type=str,         default="/predict_length",help="API endpoint.")

    dataset_group = parser.add_argument_group("dataset options")
    dataset_group.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["local", "random", "generation"],
        help="Name of the dataset to benchmark on.",
    )
    dataset_group.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Input length for each request. only in random dataset.",
    )
    dataset_group.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="Output length for each request. only in random dataset.",
    )
    
    dataset_group.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--simple-result", action="store_true")
    parser.add_argument("--trace-result", action="store_true")
    parser.add_argument("--mixed_input", action="store_true")

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives for goodput as \"KEY:VALUE\" "
        "pairs, where the key is a metric name, and the value is in "
        "milliseconds. Multiple \"KEY:VALUE\" pairs can be provided, "
        "separated by spaces. Allowed request level metric names are "
        "\"ttft\", \"tpot\", \"e2el\". For more context on the definition of "
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve")

    local_group = parser.add_argument_group("local dataset options")
    local_group.add_argument(
        "--fixed-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the local dataset.",
    )

    parser.add_argument(
        "--num-prompts-mixed",
        type=str,  # Changed from Any to str
        nargs="?",  # Changed from "+" to "?" to accept a single string argument
        help="Number of prompts to process. Format: '[val1,val2,...]'",
    )


    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral'],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        'always use the slow tokenizer. \n* '
        '"mistral" will always use the `mistral_common` tokenizer.')

    parser.add_argument(
        '--scheduling-policy',
        type=str,
        default=""
    )
    parser.add_argument(
        '--next-scheduling-policy',
        type=str,
        default=None,
        help='Next scheduling policy.'
    )
    
    args = parser.parse_args()
    main(args)
