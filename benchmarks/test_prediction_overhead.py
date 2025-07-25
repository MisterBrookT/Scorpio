#!/usr/bin/env python3
"""
Test script to measure the overhead of the prediction service at port 8001.
This script sends requests directly to the prediction service one by one and measures response times.
No need to launch vllm serve - this only tests the prediction service.
Uses async functions but still processes requests sequentially (not batched).
"""

import argparse
import time
import json
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from tqdm import tqdm

def load_prompts_from_dataset(dataset_path, num_samples=None):
    """Load prompts from the dataset JSONL file."""
    try:
        prompts = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse each line as a JSON object
                data = json.loads(line.strip())
                if 'prompt' in data:
                    # Get the prompt text
                    prompt_text = data['prompt']
                    # For testing, we're just measuring overhead, so actual length doesn't matter
                    # but maintaining the expected structure
                    prompts.append({
                        "prompt": prompt_text,
                        "actual_length": len(prompt_text),  # Just use text length as a placeholder
                        "expected_predicted_length": None   # Not needed for speed test
                    })
                
                # If we have enough samples, break
                if num_samples is not None and len(prompts) >= num_samples:
                    break
        
        return prompts
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

async def measure_prediction_service(prompts, host="localhost", port=8001, timeout=5):
    """Send requests to the prediction service one by one using async and measure performance."""
    prediction_url = f"http://{host}:{port}/predict_length"
    results = []
    
    # Create a single session for all requests
    async with aiohttp.ClientSession() as session:
        total_start_time = time.time()
        
        for prompt_data in tqdm(prompts, desc="Processing prompts"):
            prompt = prompt_data["prompt"]
            try:
                # Prepare the request payload
                payload = {"prompt": prompt}
                
                # Measure request time
                start_time = time.time()
                async with session.post(
                    url=prediction_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        prediction_data = await response.json()
                        end_time = time.time()
                        result = {
                            "success": True,
                            "latency": end_time - start_time,
                            "predicted_length": prediction_data.get("predicted_length", -1),
                            "actual_length": prompt_data["actual_length"],
                            "expected_predicted_length": prompt_data["expected_predicted_length"]
                        }
                    else:
                        end_time = time.time()
                        result = {
                            "success": False,
                            "latency": end_time - start_time,
                            "error": f"HTTP error: {response.status}"
                        }
            except Exception as e:
                result = {
                    "success": False,
                    "latency": 0.0,
                    "error": str(e)
                }
            
            results.append(result)
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
    
    return results, total_duration

def analyze_results(results, total_duration):
    """Analyze the results and calculate metrics focused on throughput and processing time."""
    successful_results = [r for r in results if r["success"]]
    success_rate = len(successful_results) / len(results) if results else 0
    
    if not successful_results:
        print("\nNo successful requests to analyze")
        return {}
    
    # Calculate latency metrics (in milliseconds)
    latencies_ms = [r["latency"] * 1000 for r in successful_results]
    
    # Calculate throughput metrics
    requests_per_second = len(successful_results) / total_duration
    avg_processing_time_sec = np.mean([r["latency"] for r in successful_results])
    
    # Print simplified results focusing on throughput and processing time
    print("\n{:=^80}".format(" Prediction Service Overhead Test Results "))
    print(f"\nOverall Performance:")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful requests: {len(successful_results)}")
    print(f"  Success rate: {success_rate*100:.2f}%")
    print(f"  Total duration: {total_duration:.2f} seconds")
    
    print(f"\nThroughput Metrics:")
    print(f"  Throughput: {requests_per_second:.2f} requests/second")
    print(f"  Mean processing time: {avg_processing_time_sec*1000:.2f} ms per request")
    
    # Create a simple results dictionary
    analysis_results = {
        "total_requests": len(results),
        "successful_requests": len(successful_results),
        "success_rate": success_rate,
        "total_duration": total_duration,
        "throughput": requests_per_second,
        "mean_processing_time_ms": avg_processing_time_sec * 1000
    }
    
    return analysis_results

async def main():
    parser = argparse.ArgumentParser(description="Test prediction service overhead")
    parser.add_argument("--host", type=str, default="localhost", help="Prediction service host")
    parser.add_argument("--port", type=int, default=8001, help="Prediction service port")
    parser.add_argument("--dataset", type=str, 
                        default="", 
                        help="Path to dataset json file with prompts to test")
    parser.add_argument("--num-samples", type=int, default=1000, 
                        help="Number of samples to use from the dataset")
    parser.add_argument("--save-results", action="store_true", help="Save results to a JSON file")
    parser.add_argument("--timeout", type=int, default=5, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Load prompts from dataset
    print(f"Loading prompts from {args.dataset}...")
    prompts = load_prompts_from_dataset(args.dataset, args.num_samples)
    if not prompts:
        print(f"Failed to load prompts from dataset {args.dataset}. Exiting.")
        return
    print(f"Loaded {len(prompts)} prompts from dataset")
    
    # Run the test
    print(f"Testing prediction service at http://{args.host}:{args.port}/predict_length")
    results, total_duration = await measure_prediction_service(
        prompts=prompts,
        host=args.host,
        port=args.port,
        timeout=args.timeout
    )
    
    # Analyze the results
    analysis_results = analyze_results(results, total_duration)
    
    # Save results if requested
    if args.save_results:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"prediction_overhead_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "test_config": {
                    "host": args.host,
                    "port": args.port,
                    "dataset": args.dataset,
                    "num_samples": args.num_samples,
                    "timeout": args.timeout
                },
                "analysis": analysis_results,
                "results": [{k: v for k, v in r.items() if k != 'error' or isinstance(v, (str, int, float, bool, type(None)))}
                           for r in results]
            }, f, indent=2)
        
        print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())