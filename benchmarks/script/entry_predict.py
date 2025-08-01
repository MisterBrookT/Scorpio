import argparse
import os
import subprocess
import sys
# Model configurations
llama8b_sharegpt_config = {
    "config_path": "MODELS/opt-125m-llama8b-sharegpt-equalwidth-numbucket100-bucketsize82-bs64-e8/usage_config.json",
    "label_group_size": 82
}
llama8b_opencoder_config = {
    "config_path": "MODELS/opt-125m-llama8b-opencoder-class-equalwidth-numbucket100-bucketsize82-b64/usage_config.json",
    "label_group_size": 82
}

llama8b_lmsys_config = {
    "config_path": "MODELS/opt-125m-llama8b-lmsys-equalwidth-numbucket100-bucketsize82-bs64-e8/usage_config.json",
    "label_group_size": 82
}

llama8b_arxiv_config = {
    "config_path": "MODELS/opt-125m-arxiv-class-equalwidth-numbucket100-bucketsize82-b64/usage_config.json",
    "label_group_size": 82
}

gemma27b_sharegpt_config = {
    "config_path": "MODELS/opt-125m-gemma27b-sharegpt-equalwidth-numbucket100-bucketsize82-bs64-e8/usage_config.json",
    "label_group_size": 82
}

gemma27b_lmsys_config = {
    "config_path": "MODELS/opt-125m-gemma27b-lmsys-equalwidth-numbucket100-bucketsize82-bs64-e8/usage_config.json",
    "label_group_size": 82
}



def get_config(model: str, dataset: str):
    configs = {
        ('8b', 'sharegpt'): llama8b_sharegpt_config,
        ('8b', 'lmsys'): llama8b_lmsys_config,
        ('8b', 'opencoder'): llama8b_opencoder_config,
        ('8b', 'arxiv'): llama8b_arxiv_config,
        ('27b', 'sharegpt'): gemma27b_sharegpt_config,
        ('27b', 'lmsys'): gemma27b_lmsys_config,
    }
    return configs.get((model, dataset))

def main():
    parser = argparse.ArgumentParser(description='Run prediction with specified model and dataset')
    parser.add_argument('--model', choices=['8b', '27b'], required=True,
                      help='Model to use for prediction')
    parser.add_argument('--dataset', choices=['lmsys', 'sharegpt', "opencoder", "arxiv"], required=True,
                      help='Dataset to use for prediction')
    
    args = parser.parse_args()
    
    config = get_config(args.model, args.dataset)
    if not config:
        raise ValueError(f"Invalid combination of model {args.model} and dataset {args.dataset}")
    
    # Create a shell script with the commands
    python_path = sys.executable  # current environment python path

    commands = [
        f"{python_path} predictor/seq_predictor/entrypoint.py \\",
        f"    --config \"{config['config_path']}\" \\",
        f"    --label-group-size {config['label_group_size']}"
    ]
    
    # Join commands with newlines and execute
    command_str = "\n".join(commands)
    print(f"Running commands:\n{command_str}")
    
    # Execute the commands using bash
    process = subprocess.run(command_str, shell=True, executable='/bin/bash')
    
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        exit(1)

if __name__ == "__main__":
    main()