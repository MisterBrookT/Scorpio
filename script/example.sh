#equal-length
python predictor/seq_predictor/trainer.py --config MODELS/opt-125m/default_config.txt --training-file ./datasets/opencoder/train.jsonl --validation-file ./datasets/opencoder/val.jsonl --test-file ./datasets/opencoder/test.jsonl --job-dir MODELS --run-id opt-125m-llama3-8b-opencoder-class-equalwidth-numbucket10-bucketsize820-b64 --batch-size 64 --label-group-size 820 --epoch 5 --check-exist --tokenizer /root/autodl-pub/models/llama3.1-8b/

# # equal-frequency
# python -m predictor.trainer --config config/config_prefill_opt_classify.txt --file ./dataset/llama3-8b-sharegpt-train-t1-s0-8192.jsonl --job-dir model --run-id opt-125m-llama3-8b-sharegpt-class-equalfrequency-numbucket100-b64 --batch-size 64 --num-bins 100 --balanced-bins --epoch 1

# # test
# python -m predictor.evaluator --config model/opt-125m-llama3-8b-sharegpt-class-equalfrequency-numbins10-b64/usage_config.json --file ./dataset/llama3-8b-sharegpt-test-t1-s0-8192_with_slo.jsonl --batch-size 64 --balanced-bins --num-bins 10