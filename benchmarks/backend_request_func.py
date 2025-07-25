import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

import aiohttp
import huggingface_hub.constants
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    engine_api_url: str  # Changed from api_url to engine_api_url
    prompt_len: int
    output_len: int
    model: str
    ttft: int
    tpot: int
    best_of: int = 1

    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    predictor_api_url: Optional[str] = None  # Changed from predictor_url to predictor_api_url
    generation_mode: bool = False
    

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    top: float = 0.0 # time of prediction
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    reject: bool = False

@dataclass
class SchedulingMetric:
    scheduler_profile: dict = field(default_factory=dict)
    
    success: bool = False

async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    engine_api_url = request_func_input.engine_api_url
    assert engine_api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."
    
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
         # --- 1. Call Prediction Server --
        prediction_url = request_func_input.predictor_api_url 
        if prediction_url is not None:
            try:
                prediction_start = time.perf_counter()
                prediction_url = request_func_input.predictor_api_url 
                pred_payload = {"prompt": request_func_input.prompt}

                # Use the session, but override timeout for this specific request
                async with session.post(url=prediction_url,
                                        json=pred_payload,
                                        timeout=aiohttp.ClientTimeout(total=5)) as pred_response:
                    if pred_response.status == 200:
                        prediction_data = await pred_response.json()
                        predicted_len = prediction_data.get("predicted_length", None)
                        assert predicted_len is not None, "some error happen"
                        output.top = time.perf_counter() - prediction_start
            except Exception as e:
            # If prediction server fails, use the original pred_output_len
                predicted_len = -1
                print(f"Prediction server error: {str(e)}. Using original prediction.")
        else:
            predicted_len = -1

        # --- 2. Call vLLM Server ---
        if not request_func_input.generation_mode:
            payload = {
                "model": request_func_input.model,
                "prompt": request_func_input.prompt,
                "temperature": 0.0,
                "best_of": request_func_input.best_of,
                "min_tokens": request_func_input.output_len-1,
                "max_tokens": request_func_input.output_len,
                "pred_output_len": int(predicted_len),  # Use the predicted length
                "ttft": request_func_input.ttft,
                "tpot": request_func_input.tpot,
                "logprobs": request_func_input.logprobs,
                "stream": True,
                "ignore_eos": request_func_input.ignore_eos,
            }
        else:
            payload = {
                "model": request_func_input.model,
                "prompt": request_func_input.prompt,
                "temperature": 1.0,
                "best_of": request_func_input.best_of,
                "max_tokens": request_func_input.output_len,
                "pred_output_len": predicted_len,  # Use the predicted length
                "ttft": request_func_input.ttft,
                "tpot": request_func_input.tpot,
                "logprobs": request_func_input.logprobs,
                "stream": True,
                "ignore_eos": request_func_input.ignore_eos,
            }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        try:
            async with session.post(url=engine_api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")

                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)
                            if data["choices"][0].get("finish_reason") == "reject":
                                output.reject = True
                                # Record the rejection
                                output.success = False
                                output.error = "Request rejected by scheduler"
                                continue 
                            
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    if output.reject:
                        pass
                    elif first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
                    output.generated_text = generated_text
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    engine_api_url = request_func_input.engine_api_url
    assert engine_api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                },
            ],
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=engine_api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv('VLLM_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
            pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(
            pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                         trust_remote_code=trust_remote_code)


def get_scheduling_metric(api_url) -> SchedulingMetric:
    output = SchedulingMetric()
    import requests

    try:
        response = requests.get(api_url).json()
        output.scheduler_profile = response['scheduler_profile']
        output.success = True
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
}
