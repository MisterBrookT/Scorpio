import math
import subprocess
llama8b_sharegpt_training_config = {
    "num_bucket": [20, 50, 100, 200, 500, 1000],
    "batch_size": 64,
    "epoch": 8,
    "model_name": "opt-125m",
    "dataset_name": "llama8b-sharegpt",
    "training_file": "/root/autodl-tmp/ada-place/datasets/llama8b-sharegpt/train.jsonl",
    "validation_file": "/root/autodl-tmp/ada-place/datasets/llama8b-sharegpt/val.jsonl",
    "test_file": "/root/autodl-tmp/ada-place/datasets/llama8b-sharegpt/test.jsonl"
}

llama8b_lmsys_training_config = {
    "num_bucket": [
        20, 50, 100, 200, 500, 1000],
    "batch_size": 64,
    "epoch": 8,
    "model_name": "opt-125m",
    "dataset_name": "llama8b-lmsys",
    "training_file": "/root/autodl-tmp/ada-place/datasets/llama8b-lmsys/train.jsonl",
    "validation_file": "/root/autodl-tmp/ada-place/datasets/llama8b-lmsys/val.jsonl",
    "test_file": "/root/autodl-tmp/ada-place/datasets/llama8b-lmsys/test.jsonl"
}

llama8b_sharegpt_extra_training_config = {
    "num_bucket": [10],
    "batch_size": 64,
    "epoch": 8,
    "model_name": "opt-125m",
    "dataset_name": "llama8b-sharegpt",
    "training_file": "/root/autodl-tmp/ada-place/datasets/llama8b-sharegpt/train.jsonl",
    "validation_file": "/root/autodl-tmp/ada-place/datasets/llama8b-sharegpt/val.jsonl",
    "test_file": "/root/autodl-tmp/ada-place/datasets/llama8b-sharegpt/test.jsonl"
}

llama8b_lmsys_extra_training_config = {
    "num_bucket": [10],
    "batch_size": 64,
    "epoch": 8,
    "model_name": "opt-125m",
    "dataset_name": "llama8b-lmsys",
    "training_file": "/root/autodl-tmp/ada-place/datasets/llama8b-lmsys/train.jsonl",
    "validation_file": "/root/autodl-tmp/ada-place/datasets/llama8b-lmsys/val.jsonl",
    "test_file": "/root/autodl-tmp/ada-place/datasets/llama8b-lmsys/test.jsonl"
}

llama70b_sharegpt_training_config = {
    "num_bucket": [
        20, 50, 100, 200, 500, 1000],
    "batch_size": 64,
    "epoch": 8,
    "model_name": "opt-125m",
    "dataset_name": "llama70b-sharegpt",
    "training_file": "/root/autodl-tmp/ada-place/datasets/llama70b-sharegpt/train.jsonl",
    "validation_file": "/root/autodl-tmp/ada-place/datasets/llama70b-sharegpt/val.jsonl",
    "test_file": "/root/autodl-tmp/ada-place/datasets/llama70b-sharegpt/test.jsonl"
}

llama70b_lmsys_training_config = {
    "num_bucket": [
        20, 50, 100, 200, 500, 1000],
    "batch_size": 64,
    "epoch": 8,
    "model_name": "opt-125m",
    "dataset_name": "llama70b-lmsys",
    "training_file": "/root/autodl-tmp/ada-place/datasets/llama70b-lmsys/train.jsonl",
    "validation_file": "/root/autodl-tmp/ada-place/datasets/llama70b-lmsys/val.jsonl",
    "test_file": "/root/autodl-tmp/ada-place/datasets/llama70b-lmsys/test.jsonl"
}

gemma27b_sharegpt_training_config = {
    "num_bucket": [10, 20, 50, 100, 200, 500, 1000],
    "batch_size": 64,
    "epoch": 8,
    "model_name": "opt-125m",
    "dataset_name": "gemma27b-sharegpt",
    "training_file": "/root/autodl-tmp/ada-place/datasets/gemma27b-sharegpt/train.jsonl",
    "validation_file": "/root/autodl-tmp/ada-place/datasets/gemma27b-sharegpt/val.jsonl",
    "test_file": "/root/autodl-tmp/ada-place/datasets/gemma27b-sharegpt/test.jsonl"
}

gemma27b_lmsys_training_config = {
    "num_bucket": [10, 20, 50, 100, 200, 500, 1000],
    "batch_size": 64,
    "epoch": 8,
    "model_name": "opt-125m",
    "dataset_name": "gemma27b-lmsys",
    "training_file": "/root/autodl-tmp/ada-place/datasets/gemma27b-lmsys/train.jsonl",
    "validation_file": "/root/autodl-tmp/ada-place/datasets/gemma27b-lmsys/val.jsonl",
    "test_file": "/root/autodl-tmp/ada-place/datasets/gemma27b-lmsys/test.jsonl"
}

def train(config, equalwidth=True):
    # equal-width
    if equalwidth:
        for num_bucket in config["num_bucket"]:
            label_group_size = math.ceil(8192 / num_bucket)
            run_id = f"{config['model_name']}-{config['dataset_name']}-equalwidth-numbucket{num_bucket}-bucketsize{label_group_size}-bs{config['batch_size']}-e{config['epoch']}"

            command = f"cd /root/autodl-tmp/ada-place/predictor/seq_predictor && python seq_predictor/trainer.py --config config/config_prefill_opt_classify.txt --training-file {config['training_file']} --validation-file {config['validation_file']} --test-file {config['test_file']} --job-dir MODELS --run-id {run_id} --batch-size {config['batch_size']} --label-group-size {label_group_size} --epoch {config['epoch']} --check-exist"
            subprocess.run(command, shell=True)
    # equal-frequency
    else:
        for num_bucket in config["num_bucket"]:
            run_id = f"{config['model_name']}-{config['dataset_name']}-equalfreq-numbucket{num_bucket}-bs{config['batch_size']}-e{config['epoch']}"

            command = f"cd /root/autodl-tmp/ada-place/predictor/seq_predictor && python seq_predictor/trainer.py --config config/config_prefill_opt_classify.txt --training-file {config['training_file']} --validation-file {config['validation_file']} --test-file {config['test_file']} --job-dir MODELS --run-id {run_id} --batch-size {config['batch_size']} --epoch {config['epoch']} --balanced-bins --num-bins {num_bucket} --check-exist"
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    # train(llama8b_sharegpt_training_config, equalwidth=True)
    # train(llama8b_sharegpt_training_config, equalwidth=False)
    # train(llama8b_lmsys_training_config, equalwidth=True)
    # train(llama8b_lmsys_training_config, equalwidth=False)
    # train(llama70b_sharegpt_training_config, equalwidth=True)
    # train(llama3_70b_sharegpt_training_config, equalwidth=False)
    # train(llama70b_lmsys_training_config, equalwidth=True)
    # train(llama3_70b_lmsys_training_config, equalwidth=False)
    # train(gemma27b_sharegpt_training_config, equalwidth=True)
    # train(gemma27b_sharegpt_training_config, equalwidth=False)
    # train(gemma27b_lmsys_training_config, equalwidth=True)
    # train(gemma27b_lmsys_training_config, equalwidth=False)

    train(llama8b_sharegpt_extra_training_config, equalwidth=True)
    train(llama8b_lmsys_extra_training_config, equalwidth=True)
    train(llama8b_sharegpt_extra_training_config, equalwidth=False)
    train(llama8b_lmsys_extra_training_config, equalwidth=False)
