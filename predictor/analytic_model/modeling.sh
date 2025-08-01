#!/bin/bash

# Run prefill modeling
# python modeling_prefill.py \
#     --model_name "llama8b-sharegpt" \
#     --data_path "benchmarks/result/pre_exp/llama8b-sharegpt_decode_result_different_batch_size.json" \

# # Run decode modeling
# python modeling_decode.py \
#     --model_name "llama8b-sharegpt" \
#     --data_path "benchmarks/result/pre_exp/llama8b-sharegpt_decode_result_different_batch_size.json"

# Run prefill modeling for Gemma 27B
# python predictor/analytic_model/modeling_prefill.py \
#     --model_name "gemma27b-sharegpt-prefill" \
#     --data_path "predictor/analytic_model/profiled_result/gemma27b-sharegpt-prefill.jsonl" \

# # Run decode modeling for Gemma 27B
# python predictor/analytic_model/modeling_decode.py \
#     --model_name "gemma27b-sharegpt-decode" \
#     --data_path "predictor/analytic_model/profiled_result/gemma27b-sharegpt-decode.jsonl"


# Run prefill modeling for Gemma 27B LMSYS
# python predictor/analytic_model/modeling_prefill.py \
#     --model_name "gemma27b-lmsys-prefill" \
#     --data_path "predictor/analytic_model/profiled_result/gemma27b-lmsys-prefill.jsonl"

# # Run decode modeling for Gemma 27B LMSYS
# python predictor/analytic_model/modeling_decode.py \
#     --model_name "gemma27b-lmsys-decode" \
#     --data_path "predictor/analytic_model/profiled_result/gemma27b-lmsys-decode.jsonl"

# Run prefill modeling for Llama 8B 
python predictor/analytic_model/modeling_prefill.py \
    --model_name "llama8b-arxiv-prefill" \
    --data_path "predictor/analytic_model/profiled_result/llama8b-arxiv-prefill.jsonl" \

# Run decode modeling for Llama 8B 
python predictor/analytic_model/modeling_decode.py \
    --model_name "llama8b-arxiv-decode" \
    --data_path "predictor/analytic_model/profiled_result/llama8b-arxiv-decode.jsonl"