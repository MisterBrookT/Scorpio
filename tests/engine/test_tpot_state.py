import pytest
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from transformers import AutoTokenizer
import random
from vllm import LLMEngine, RequestOutput
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple
import itertools

# MKL_SERVICE_FORCE_INTEL=1 python -m pytest tests/engine/test_tpot_state.py -v

@pytest.fixture
def model_path():
    return "facebook/opt-125m"  # Example model path

@pytest.fixture
def engine_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Use fixed arguments for testing
    args = parser.parse_args([
        "--model", "facebook/opt-125m",
        "--max-model-len", "2048",
        "--max-num-seqs", "256",
        "--scheduling-policy", "fcfs_rej",
        "--load-format", "dummy",
        "--enforce-eager",
        "--prefill-prediction-model-path", "analytic_model/config/llama8b-sharegpt-prefill.json",
    ])
    return EngineArgs.from_cli_args(args)

@pytest.fixture
def engine(engine_args):
    return LLMEngine.from_engine_args(engine_args)

def create_test_prompts(input_len, output_len, model_path, ttft, tpot):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    candidate_ids = [
        random.randint(0, vocab_size - 1)
        for _ in range(input_len)
    ]

    candidate_prompt = tokenizer.decode(candidate_ids)
    return candidate_prompt, SamplingParams(temperature=0, top_p=1.0, ignore_eos=True, min_tokens=output_len, max_tokens=output_len, ttft=ttft, tpot=tpot)

def test_add_delete_request(engine, model_path):
    """Test the TPOT state functionality."""
    # test in normal mode, can get the correct tpot state   
    prompt1, params1 = create_test_prompts(input_len=128, output_len=2, model_path=model_path, ttft=1000, tpot=50)
    prompt2, params2 = create_test_prompts(input_len=256, output_len=2, model_path=model_path, ttft=1000, tpot=50)
    prompt3, params3 = create_test_prompts(input_len=128, output_len=10, model_path=model_path, ttft=1000, tpot=100)
    prompt4, params4 = create_test_prompts(input_len=256, output_len=10, model_path=model_path, ttft=1000, tpot=200)

    engine.add_request(request_id="request1", prompt=prompt1, params=params1)
    engine.add_request(request_id="request2", prompt=prompt2, params=params2)
    engine.add_request(request_id="request3", prompt=prompt3, params=params3)
    engine.add_request(request_id="request4", prompt=prompt4, params=params4)
    engine.step()
    assert engine.scheduler[0].tpot_state.get_lowest_tpot() == 50
    assert engine.scheduler[0].tpot_state.unique_active_tpots == sorted([50, 100, 200])

    engine.step()
    assert engine.scheduler[0].tpot_state.get_lowest_tpot() == 100
    assert engine.scheduler[0].tpot_state.unique_active_tpots == sorted([100, 200])

    # test add request with different tpot
    prompt5, params5 = create_test_prompts(input_len=128, output_len=10, model_path=model_path, ttft=1000, tpot=20)
    prompt6, params6 = create_test_prompts(input_len=256, output_len=10, model_path=model_path, ttft=1000, tpot=20)

    engine.add_request(request_id="request5", prompt=prompt5, params=params5)
    engine.add_request(request_id="request6", prompt=prompt6, params=params6)
    engine.step()
    assert engine.scheduler[0].tpot_state.get_lowest_tpot() == 20
    assert engine.scheduler[0].tpot_state.unique_active_tpots == sorted([20, 100, 200])
    
    # test with rejection, nothing changes
    prompt7, params7 = create_test_prompts(input_len=128, output_len=10, model_path=model_path, ttft=-1000, tpot=10)
    engine.add_request(request_id="request7", prompt=prompt7, params=params7)
    engine.step()
    assert engine.scheduler[0].tpot_state.get_lowest_tpot() == 20
    assert engine.scheduler[0].tpot_state.unique_active_tpots == sorted([20, 100, 200])


def test_virtual_batch_size(engine, model_path):
    """Test the TPOT state functionality."""
    # test in normal mode, can get the correct tpot state   
    prompt1, params1 = create_test_prompts(input_len=128, output_len=2, model_path=model_path, ttft=1000, tpot=50)
    prompt2, params2 = create_test_prompts(input_len=256, output_len=2, model_path=model_path, ttft=1000, tpot=50)
    prompt3, params3 = create_test_prompts(input_len=128, output_len=10, model_path=model_path, ttft=1000, tpot=100)
    prompt4, params4 = create_test_prompts(input_len=256, output_len=10, model_path=model_path, ttft=1000, tpot=200)

    engine.add_request(request_id="request1", prompt=prompt1, params=params1)
    engine.add_request(request_id="request2", prompt=prompt2, params=params2)
    engine.add_request(request_id="request3", prompt=prompt3, params=params3)
    engine.add_request(request_id="request4", prompt=prompt4, params=params4)
    engine.step()
    assert engine.scheduler[0].tpot_state.get_virtual_batch_size() == 50/50 + 50/50 + 50/100 + 50/200

    engine.step()
    assert engine.scheduler[0].tpot_state.get_virtual_batch_size() == 100/100 + 100/200


