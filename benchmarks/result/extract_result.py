import json
import os
from pathlib import Path
import csv
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_all_json_files(directory):
    results = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    for file_path in directory_path.glob("**/*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append({
                    'file_name': str(file_path),
                    'data': data
                })
        except json.JSONDecodeError as e:
            print(f"Error reading {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error reading {file_path}: {e}")
    
    return results

def process_results(json_results):
    results_list = []
    
    for result in json_results:

        try:
            filename = os.path.basename(result['file_name'])
            data = result['data']
            policy = data['policy']
            duration = data['duration']
            request_rate = data['request_rate']
            goodput = round(data['request_goodput'], 2)
            good_completion = data['good_completion']
            completed = data['completed']
            slo_adherence_ratio = data['slo_adherence_ratio']
            schedule_time = data['scheduler_profile']['schedule_time']
            scheduler_running_other_time = data['scheduler_profile'].get('scheduler_running_other_time', -100)
            credit_time = data['scheduler_profile'].get('scheduler_running_credit_time', -100)
            reject_time = data['scheduler_profile'].get('reject_time', -100)
            reorder_time = data['scheduler_profile'].get('reorder_time', -100)
            admission_control_time = data['scheduler_profile'].get('admission_control_time', -100)
            prefill_schedule_time = data['scheduler_profile'].get('prefill_schedule_time', -100)

            ttft_violation = data['ttft_violation']
            tpot_violation = data['tpot_violation']
            both_violation = data['both_violation']
            rejected = data['rejected']
        except Exception as e:
            print(f"Error processing {result['file_name']}: {e}")
            continue
        results_list.append({
            'filename': filename,
            'policy': policy,
            'duration': duration,
            'request_rate': request_rate,
            'goodput': goodput,
            'good_completion': good_completion,
            'completed': completed,
            'slo_adherence': slo_adherence_ratio,
            'schedule_time': schedule_time,
            'prefill_schedule_time': prefill_schedule_time,
            'credit_time': credit_time,
            'reject_time': reject_time,
            'reorder_time': reorder_time,
            'admission_control_time': admission_control_time,
            'scheduler_running_other_time': scheduler_running_other_time,
            'ttft_violation': ttft_violation,
            'tpot_violation': tpot_violation,
            'both_violation': both_violation,
            'rejected': rejected
        })
    
    return results_list

def filter_top_results(results_list, top_n=3):
    grouped_results = {}
    for result in results_list:
        request_rate = result['request_rate']
        if request_rate not in grouped_results:
            grouped_results[request_rate] = []
        grouped_results[request_rate].append(result)
    
    filtered_results = []
    for request_rate, results in grouped_results.items():
        sorted_results = sorted(results, key=lambda x: x['slo_adherence'], reverse=True)
        filtered_results.extend(sorted_results[:top_n])
    
    return filtered_results

def analyze_parameters(csv_path):
    """
    Analyze the CSV file to find the best performing policies based on goodput and slo_adherence.
    Uses average metrics across all runs.
    
    Args:
        csv_path (str): Path to the CSV file containing the results
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter for average data
    avg_df = df[df['run'] == 'avg']
    
    # Group by policy and calculate mean of goodput and slo_adherence
    policy_means = avg_df.groupby('policy').agg({
        'goodput': 'mean',
        'slo_adherence': 'mean'
    }).reset_index()
    
    # Sort by goodput in descending order
    policy_means = policy_means.sort_values('goodput', ascending=False)
    
    # Print the results
    print("\nAverage goodput by policy (sorted from highest to lowest):")
    print("=" * 50)
    for _, row in policy_means.iterrows():
        print(f"{row['policy']}: {row['goodput']:.2f}")
    
    print("\nAverage slo_adherence by policy (sorted from highest to lowest):")
    print("=" * 50)
    for _, row in policy_means.iterrows():
        print(f"{row['policy']}: {row['slo_adherence']:.2f}")

def create_bar_plots(df, output_dir):
    """
    Create bar plots for each run (r1, r2, etc.) and average, with four subplots:
    1. Goodput by policy for each QPS
    2. SLO adherence by policy for each QPS
    3. Schedule time by policy for each QPS
    4. Per-request schedule time by policy for each QPS
    
    Args:
        df (pd.DataFrame): DataFrame containing the results
        output_dir (str): Directory to save the plots
    """
    # Set the style
    plt.style.use('default')
    
    # Get unique runs including 'avg'
    runs = sorted(df['run'].unique())
    
    for run in runs:
        print(f"\nProcessing run: {run}")
        # Create a proper copy of the DataFrame for this run
        run_df = df[df['run'] == run].copy()
        
        # Get unique policies and request rates for this run
        policies = run_df['policy'].unique()
        request_rates = sorted(run_df['request_rate'].unique())
        
        # Calculate per-request schedule time
        run_df['per_request_schedule_time'] = run_df['schedule_time'] / run_df['completed']
        
        # Create a single figure with four subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        title = f'Performance Metrics by Policy for Different QPS - {run.upper()}'
        if run == 'avg':
            title = 'Average Performance Metrics by Policy for Different QPS'
        fig.suptitle(title, fontsize=16, y=0.95)
        
        x = np.arange(len(request_rates))
        width = 0.8 / len(policies)  # Adjust bar width based on number of policies
        
        # Plot 1: Goodput
        for i, policy in enumerate(policies):
            policy_data = run_df[run_df['policy'] == policy]
            values = [policy_data[policy_data['request_rate'] == rate]['goodput'].mean() for rate in request_rates]
            axes[0, 0].bar(x + i * width - 0.4 + width/2, values, width, label=policy)
        
        axes[0, 0].set_title('Goodput by Policy')
        axes[0, 0].set_xlabel('Request Rate (QPS)')
        axes[0, 0].set_ylabel('Goodput')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(request_rates, rotation=45)
        
        # Plot 2: SLO Adherence
        for i, policy in enumerate(policies):
            policy_data = run_df[run_df['policy'] == policy]
            values = [policy_data[policy_data['request_rate'] == rate]['slo_adherence'].mean() for rate in request_rates]
            axes[0, 1].bar(x + i * width - 0.4 + width/2, values, width, label=policy)
        
        axes[0, 1].set_title('SLO Adherence by Policy')
        axes[0, 1].set_xlabel('Request Rate (QPS)')
        axes[0, 1].set_ylabel('SLO Adherence')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(request_rates, rotation=45)
        
        # Plot 3: Schedule Time
        for i, policy in enumerate(policies):
            policy_data = run_df[run_df['policy'] == policy]
            values = [policy_data[policy_data['request_rate'] == rate]['schedule_time'].mean() for rate in request_rates]
            axes[1, 0].bar(x + i * width - 0.4 + width/2, values, width, label=policy)
        
        axes[1, 0].set_title('Schedule Time by Policy')
        axes[1, 0].set_xlabel('Request Rate (QPS)')
        axes[1, 0].set_ylabel('Schedule Time (s)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(request_rates, rotation=45)
        
        # Plot 4: Per-Request Schedule Time
        for i, policy in enumerate(policies):
            policy_data = run_df[run_df['policy'] == policy]
            values = [policy_data[policy_data['request_rate'] == rate]['per_request_schedule_time'].mean() for rate in request_rates]
            axes[1, 1].bar(x + i * width - 0.4 + width/2, values, width, label=policy)
        
        axes[1, 1].set_title('Per-Request Schedule Time by Policy')
        axes[1, 1].set_xlabel('Request Rate (QPS)')
        axes[1, 1].set_ylabel('Per-Request Schedule Time (s)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(request_rates, rotation=45)
        
        # Add a single legend for all subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Policy', bbox_to_anchor=(1.02, 0.5), loc='center left')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the figure
        plot_name = f'{run}_summary.png' if run != 'avg' else 'average_summary.png'
        plot_path = os.path.join(output_dir, plot_name)
        print(f"Saving plot to: {plot_path}")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Process JSON result files and generate analysis CSV')
    parser.add_argument('--input-dir', type=str, help='Directory containing JSON result files')
    args = parser.parse_args()
    
    print(f"\nProcessing results from directory: {args.input_dir}")
    
    # Create output directory for plots if it doesn't exist
    os.makedirs(args.input_dir, exist_ok=True)
    
    # Read and process JSON files
    json_results = read_all_json_files(args.input_dir)
    results_list = process_results(json_results)
    
    # Create DataFrame for sorting and analysis
    df = pd.DataFrame(results_list)
    
    # Add run number for sorting
    df['run'] = df['filename'].apply(lambda x: x.split('_')[0])
    
    # Sort by filename and request_rate
    df = df.sort_values(['filename', 'request_rate'])
    
    # Calculate averages across runs for each policy and QPS
    avg_df = df.groupby(['policy', 'request_rate']).agg({
        'goodput': 'mean',
        'slo_adherence': 'mean',
        'schedule_time': 'mean',
        'completed': 'mean',
        'good_completion': 'mean',
        'ttft_violation': 'mean',
        'tpot_violation': 'mean',
        'both_violation': 'mean',
        'rejected': 'mean',
        'prefill_schedule_time': 'mean',
        'credit_time': 'mean',
        'reject_time': 'mean',
        'reorder_time': 'mean',
        'admission_control_time': 'mean',
        'scheduler_running_other_time': 'mean'
    }).reset_index()
    
    # Add run column for average data
    avg_df['run'] = 'avg'
    avg_df['filename'] = 'average'
    
    # Combine original and average data
    combined_df = pd.concat([df, avg_df], ignore_index=True)
    
    # Write results to CSV in the input directory
    csv_path = os.path.join(args.input_dir, 'summary.csv')
    print(f"\nSaving summary CSV to: {csv_path}")
    combined_df.to_csv(csv_path, index=False)
    
    # Generate plots
    print("\nGenerating plots...")
    create_bar_plots(combined_df, args.input_dir)
    
    # Analyze parameters after generating CSV
    analyze_parameters(csv_path)

if __name__ == "__main__":
    main()
