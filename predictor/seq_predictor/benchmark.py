from transformers import AutoTokenizer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import statistics
from pathlib import Path

from predictor.seq_predictor.config import PrefillPredictorConfig
from predictor.seq_predictor.model.prefill_predictor import prefill_predictor_model
from predictor.seq_predictor.utils.model_loader import set_default_torch_dtype
from predictor.seq_predictor.evaluator import EvalDataset



def parse_args():
    parser = ArgumentParser("Sequence Length Predictor Benchmark")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the usage_config.json file in the model output directory")
    parser.add_argument("--file", type=str, required=True,
                        help="Path to the test data file")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--label-max-length", type=int, default=8192, 
                                  help="Maximum token length to consider (context length for Llama-3)")
    # Organize binning strategies into groups
    equal_width_group = parser.add_argument_group('Equal-width binning', 
                                                'Strategy that divides the token length range into equal-sized bins')
    equal_width_group.add_argument("--label-group-size", type=int, default=1,
                                  help="Size of each bin in tokens")
    
    equal_freq_group = parser.add_argument_group('Equal-frequency binning',
                                               'Strategy that creates bins with approximately equal number of samples')
    equal_freq_group.add_argument("--balanced-bins", action='store_true', 
                               help="Enable balanced binning (equal number of samples per bin)")
    equal_freq_group.add_argument("--num-bins", type=int, default=100, 
                                help="Number of bins to use when balanced-bins is enabled")
    
    # Benchmark-specific parameters
    benchmark_group = parser.add_argument_group('Benchmark settings',
                                             'Parameters controlling the benchmark execution')
    benchmark_group.add_argument("--min-batch-size", type=int, default=1,
                              help="Minimum batch size to benchmark")
    benchmark_group.add_argument("--max-batch-size", type=int, default=256,
                              help="Maximum batch size to benchmark")
    benchmark_group.add_argument("--trials", type=int, default=5,
                               help="Number of trials to run for each batch size")
    benchmark_group.add_argument("--warmup-trials", type=int, default=2,
                               help="Number of warm-up trials before actual benchmarking")
    benchmark_group.add_argument("--max-samples", type=int, default=1000,
                               help="Maximum number of samples to use for benchmarking")
    
    return parser.parse_args()


def run_benchmark_for_batch_size(predictor, dataset, batch_size, num_trials, warmup_trials, max_samples):
    """Run benchmarking for a specific batch size"""
    # Limit the dataset size to max_samples
    dataset_subset = torch.utils.data.Subset(
        dataset,
        list(range(min(len(dataset), max_samples)))
    )
    
    # Create a dataloader with the specified batch size
    dataloader = DataLoader(
        dataset_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,  # Use 1 worker to minimize overhead
        pin_memory=True
    )
    
    # Record timing stats
    timing_stats = []
    
    # Run warm-up trials
    for _ in range(warmup_trials):
        for prompt, _, _, _ in dataloader:
            prompt = list(prompt)
            
            encoded_inputs = predictor.tokenizer(
                prompt,
                max_length=2048,  # Using a reasonable default
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded_inputs['input_ids'].to("cuda:0")
            attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
            
            # Dummy forward pass
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                _ = predictor(input_ids, attention_mask)
                
    # Measure actual performance
    for trial in range(num_trials):
        batch_times = []
        samples_processed = 0
        
        for prompt, _, _, _ in dataloader:
            prompt = list(prompt)
            
            encoded_inputs = predictor.tokenizer(
                prompt,
                max_length=2048,  # Using a reasonable default
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded_inputs['input_ids'].to("cuda:0")
            attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
            
            # Measure inference time
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                _ = predictor(input_ids, attention_mask)
                
            torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = end_time - start_time
            batch_times.append(batch_time)
            samples_processed += len(prompt)
        
        # Calculate average time per sample for this trial
        if batch_times:
            avg_time_per_sample = sum(batch_times) / samples_processed
            timing_stats.append(avg_time_per_sample)
    
    # Calculate statistics
    avg_time = statistics.mean(timing_stats)
    std_time = statistics.stdev(timing_stats) if len(timing_stats) > 1 else 0
    throughput = 1.0 / avg_time  # samples per second
    
    return {
        "batch_size": batch_size,
        "avg_time_per_sample": avg_time,
        "std_time": std_time,
        "throughput": throughput,
        "raw_times": timing_stats
    }


def run_benchmark(args):
    """Run the benchmark across all batch sizes"""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load model config
    config = PrefillPredictorConfig.from_json(args.config)
    print(f"Loaded config from {args.config}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Loaded tokenizer from {args.tokenizer}")
    
    # Load model
    with set_default_torch_dtype(torch.float32):
        with torch.device('cuda'):
            predictor = prefill_predictor_model(
                pred_model=config.model.path,
                num_labels=config.model.num_labels,
                mtype=config.model.mtype,
                activation=config.model.activation,
                max_length=config.model.max_length,
                max_batch_size=config.model.max_batch_size,
                tokenizer_name=config.model.pred_model,
            )
    predictor.model = predictor.model.to("cuda:0")
    predictor.model.eval()  # Set model to evaluation mode
    print(f"Loaded model from {config.model.path}")
    
    # Load dataset
    dataset_path = args.file
    if dataset_path.endswith('.csv'):
        eval_dataset = EvalDataset.from_csv(
            dataset_path,
            tokenizer,
            max_length=config.model.max_length,
            label_max_length=args.label_max_length,
            label_group_size=args.label_group_size
        )
    else:
        raise ValueError("Only CSV files are supported for benchmarking")
    
    # Convert to balanced binning if requested
    if args.balanced_bins:
        # We need to recreate the dataset with balanced bins
        data = eval_dataset.data  # Get the data from the original dataset
        eval_dataset = EvalDataset(
            data,
            tokenizer, 
            max_length=config.model.max_length, 
            label_max_length=args.label_max_length,
            label_group_size=args.label_group_size,
            balanced_bins=True,
            num_bins=args.num_bins
        )
        print(f"Using balanced binning with {args.num_bins} bins")
    
    # Use sequential batch sizes with interval of 1 instead of powers of 2
    batch_sizes = list(range(args.min_batch_size, args.max_batch_size + 1))
    
    # For large ranges, we might want to sample to keep the benchmark time reasonable
    if len(batch_sizes) > 30:  # If more than 30 different batch sizes
        # Take more samples at smaller batch sizes where differences matter more
        small_batch_sizes = list(range(args.min_batch_size, min(32, args.max_batch_size + 1)))
        
        # For larger batch sizes, use a bigger interval to reduce total benchmarking time
        if args.max_batch_size > 32:
            large_batch_interval = max(1, (args.max_batch_size - 32) // 20)  # Aim for about 20 samples in the larger range
            large_batch_sizes = list(range(32, args.max_batch_size + 1, large_batch_interval))
            batch_sizes = small_batch_sizes + large_batch_sizes
    
    batch_sizes = sorted(list(set(batch_sizes)))  # Remove any duplicates and sort
    
    print(f"Running benchmark with batch sizes: {batch_sizes}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks for each batch size
    results = []
    
    for batch_size in tqdm(batch_sizes, desc="Benchmarking batch sizes"):
        try:
            result = run_benchmark_for_batch_size(
                predictor=predictor,
                dataset=eval_dataset,
                batch_size=batch_size,
                num_trials=args.trials,
                warmup_trials=args.warmup_trials,
                max_samples=args.max_samples
            )
            results.append(result)
            print(f"Batch size {batch_size}: {result['avg_time_per_sample']*1000:.2f} ms per sample, "
                  f"throughput: {result['throughput']:.2f} samples/sec")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Batch size {batch_size} exceeded available GPU memory. Stopping benchmark.")
                break
            else:
                raise
    
    # Save results to CSV
    model_name = os.path.basename(os.path.dirname(args.config))
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"benchmark_{model_name}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Benchmark results saved to {csv_path}")
    
    # Plot the results
    plot_benchmark_results(results, model_name, args.output_dir)


def plot_benchmark_results(results, model_name, output_dir):
    """Plot the benchmark results"""
    # Extract data for plotting
    batch_sizes = [r["batch_size"] for r in results]
    avg_times = [r["avg_time_per_sample"] * 1000 for r in results]  # Convert to ms
    throughputs = [r["throughput"] for r in results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot average time per sample
    ax1.plot(batch_sizes, avg_times, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Average Time per Sample (ms)")
    ax1.set_title("Inference Time vs Batch Size")
    ax1.grid(True)
    
    # Add error bars
    std_times = [r["std_time"] * 1000 for r in results]  # Convert to ms
    ax1.errorbar(batch_sizes, avg_times, yerr=std_times, fmt='o', capsize=5)
    
    # Plot throughput
    ax2.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Throughput (samples/second)")
    ax2.set_title("Throughput vs Batch Size")
    ax2.grid(True)
    
    # Set overall title
    plt.suptitle(f"Inference Performance Benchmark - {model_name}", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(output_dir, f"benchmark_{model_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Benchmark plot saved to {fig_path}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)