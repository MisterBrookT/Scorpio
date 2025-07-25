#equal-length
python predictor/trainer.py --config config/config_prefill_opt_classify.txt --file ./dataset/llama3-8b-sharegpt-train-t1-s0-8192.jsonl --job-dir model --run-id opt-125m-llama3-8b-sharegpt-class-equalwidth-numbucket100-bucketsize82-b64 --batch-size 64 --label-group-size 82 --epoch 10

# test
python predictor/evaluator.py --config model/opt-125m-llama3-8b-sharegpt-class-trainbucket82-b64/usage_config.json --file ./dataset/llama3-8b-sharegpt-test-t1-s0-8192_with_slo.jsonl --label-group-size 82 --batch-size 64


# equal-frequency
python -m predictor.trainer --config config/config_prefill_opt_classify.txt --file ./dataset/llama3-8b-sharegpt-train-t1-s0-8192.jsonl --job-dir model --run-id opt-125m-llama3-8b-sharegpt-class-equalfrequency-numbucket100-b64 --batch-size 64 --num-bins 100 --balanced-bins --epoch 1

# test
python -m predictor.evaluator --config model/opt-125m-llama3-8b-sharegpt-class-equalfrequency-numbins10-b64/usage_config.json --file ./dataset/llama3-8b-sharegpt-test-t1-s0-8192_with_slo.jsonl --batch-size 64 --balanced-bins --num-bins 10