#!/usr/bin/env python3

import json
import subprocess
import time
import os
import sys
import signal
import requests
from pathlib import Path
import argparse
import itertools
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from contextlib import contextmanager


def json2args(json_data):
    """Convert a JSON object to command line arguments string."""
    args = []
    for key, value in json_data.items():
        args.append(f"--{key.replace('_', '-')} {value}")
    return " ".join(args)


def wait_for_server(timeout=120):
    """Wait for vllm server to start, return True if server started, False if timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            requests.post('http://localhost:8000/v1/completions')
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(5)
    return False


def kill_gpu_processes():
    """Kill GPU-related processes and clean up."""
    try:
        # Show running processes
        subprocess.run("ps -aux", shell=True)
        
        # Kill processes using port 8000
        subprocess.run("lsof -t -i:8000 | xargs -r kill -9", shell=True)
        
        # Kill all python processes
        subprocess.run("pgrep python3 | xargs -r kill -9", shell=True)
 
        # Wait until GPU memory usage is smaller than 1GB
        while True:
            result = subprocess.run(
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1",
                shell=True, text=True, capture_output=True
            )
            try:
                memory_used = int(result.stdout.strip())
                if memory_used < 2000:
                    break
            except ValueError:
                break
            time.sleep(1)
        
        # Remove vllm config file
        vllm_config = Path.home() / ".config" / "vllm"
        if vllm_config.exists():
            subprocess.run(f"rm -rf {vllm_config}", shell=True)
            
    except Exception as e:
        print(f"Error cleaning up GPU processes: {e}")


def check_gpus():
    """Check for available GPUs."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        return result.returncode == 0
    except:
        return False


# assume only one server command
def filter_existed_client_commands(test: Dict, check_exist: bool) -> List[Dict]:
    existed_combinations = []
    varying_params:Dict = test['varying_parameters']
    fixed_params: Dict = test['fixed_parameters']
        
    # server for-loop
    scheduling_policy_list = varying_params['share']['scheduling_policy']
    request_rate_list = varying_params['client']['request_rate']

    for request_rate in request_rate_list:
        for scheduling_policy in scheduling_policy_list:
            if check_exist:
                test_name= f"{test['test_name']}_{scheduling_policy}_qps_{request_rate}.json"
                result_dir = fixed_params['client']['result_dir']
                # Check if the result file already exists
                result_file_path = os.path.join(result_dir, test_name)
                if not os.path.exists(result_file_path):
                    existed_combinations.append(f"{request_rate}-{scheduling_policy}")
            else:
                existed_combinations.append(f"{request_rate}-{scheduling_policy}")
    return existed_combinations


def run_extract_result(result_dir: str, test_name: str):
    """Run extract_result.py to generate CSV from JSON results."""
    extract_cmd = f"python3 benchmarks/result/extract_result.py --input-dir {result_dir}"
    print(f"Running extract_result.py: {extract_cmd}")
    subprocess.run(extract_cmd, shell=True)


def run_client_test(client_command: str, result_dir: str, test_name: str):
    """Run a single client test and generate analysis CSV."""
    print(f"Client Command {client_command}")
    subprocess.run(client_command, shell=True)
    # Run extract_result.py after the test completes
    run_extract_result(result_dir, test_name)


def commands_from_json(test: Dict, args) -> Dict:
    varying_params: Dict = test['varying_parameters']
    fixed_params: Dict = test['fixed_parameters']
    

    # server for-loop
    scheduling_policy_list = varying_params['share']['scheduling_policy']
    request_rate_list = varying_params['client']['request_rate']
    existed_qps_policy_combinations = filter_existed_client_commands(test, args.check_exist)

    total_scheduling_policy_list = [pair.split('-')[1] for pair in existed_qps_policy_combinations]

    commands = {}
    for request_rate in request_rate_list:
        server_params = {**fixed_params['server'], **fixed_params['share'], "scheduling_policy": " ".join(total_scheduling_policy_list)}
        server_args = json2args(server_params)
        server_command = f"python3 -m vllm.entrypoints.openai.api_server {server_args}"

        # Check if the server command already exists in the commands dictionary
        if server_command not in commands:
            commands[server_command] = [] 
        # client command for-loop
        for scheduling_policy in scheduling_policy_list:
            client_params = {**fixed_params['client'],**fixed_params['share'], 'request_rate': request_rate, "scheduling_policy": scheduling_policy}
            client_args = json2args(client_params)
            test_name= f"{test['test_name']}_{scheduling_policy}_qps_{request_rate}.json"
            result_dir = client_params['result_dir']
            # Check if the result file already exists
            result_file_path = os.path.join(result_dir, test_name)
            if args.check_exist:
                if os.path.exists(result_file_path):
                    print(f"Result file {result_file_path} already exists, skipping this test")
                    continue

            client_command = (
                        f"python3 benchmarks/benchmark.py "
                        f"--save-result "
                        f"--result-filename {test_name} "
                        f"--predictor-host {fixed_params['client'].get('predictor_host', 'localhost')} "
                        f"--predictor-port {fixed_params['client'].get('predictor_port', 8001)} "
                        f"{client_args}"
                    )
            commands[server_command].append((client_command, result_dir, test['test_name']))
    return commands


@contextmanager
def vllm_server_context(server_command: str):
    """Context manager for vLLM server."""

    print(f"Starting server with command: {server_command}")
    
    server_process = subprocess.Popen(server_command, shell=True)
    try:
        if wait_for_server():
            print("\nvllm server is up and running.")
            yield True
        else:
            print("\nvllm failed to start within the timeout period.")
            yield False
    finally:
        print("Cleaning up server resources...")
        kill_gpu_processes()


def main():
    """Main entry point."""
    if not check_gpus():
        print("No GPUs available. Exiting.")
        return
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="benchmarks/online_exp/config/end2end/sharegpt.json", 
                        help='Path to the json config file')
    
    parser.add_argument('--check-exist', action='store_true', help='Check if result file exists before running the test')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        tests = json.load(f)
    # Create and run test commands

    all_commands = {}
    for i, test in enumerate(tests):
        commands = commands_from_json(test, args)
        all_commands[i] = commands

    with open("all_commands.json", "w") as f:
        json.dump(all_commands, f)

    # run all commands
    for i, commands in all_commands.items():
        for server_command, client_commands in commands.items():
            if not client_commands:
                continue
            with vllm_server_context(server_command) as server_started:    
                for client_command, result_dir, test_name in client_commands:
                    if server_started:
                        run_client_test(client_command, result_dir, test_name)
                    



if __name__ == "__main__":
    # kill_gpu_processes()
    main()