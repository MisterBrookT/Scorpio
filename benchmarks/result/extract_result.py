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
            schedule_time = data['scheduler_profile'].get('schedule_time', -100)
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
    Create one figure with four subplots:
    1. Goodput by policy across different QPS (grouped bars)
    2. SLO adherence by policy across different QPS (grouped bars)
    3. Average goodput by policy (across runs and QPS)
    4. Average SLO adherence by policy (across runs and QPS)
    
    Args:
        df (pd.DataFrame): DataFrame containing the results
        output_dir (str): Directory to save the plots
    """
    # Set the style
    plt.style.use('default')

    # Prepare aggregated data across runs (exclude 'avg' rows)
    per_qps_df = df[df['run'] != 'avg'].copy()
    # Group by policy and request_rate to average across runs
    per_qps_grouped = per_qps_df.groupby(['policy', 'request_rate']).agg({
        'goodput': 'mean',
        'slo_adherence': 'mean'
    }).reset_index()

    # Prepare average per policy (use avg rows if present; fallback to overall mean)
    avg_rows = df[df['run'] == 'avg'].copy()
    if avg_rows.empty:
        avg_rows = df.groupby('policy').agg({
            'goodput': 'mean',
            'slo_adherence': 'mean'
        }).reset_index()

    # Get sorted lists
    policies = sorted(per_qps_grouped['policy'].unique())
    request_rates = sorted(per_qps_grouped['request_rate'].dropna().unique())

    # Create a figure with four subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Policy Performance Overview', fontsize=16, y=0.95)

    # Subplot 1: Goodput by policy across QPS
    x = np.arange(len(request_rates))
    width = 0.8 / max(len(policies), 1)
    for i, policy in enumerate(policies):
        policy_data = per_qps_grouped[per_qps_grouped['policy'] == policy]
        values = [policy_data[policy_data['request_rate'] == rate]['goodput'].mean() for rate in request_rates]
        axes[0, 0].bar(x + i * width - 0.4 + width/2, values, width, label=policy)
    axes[0, 0].set_title('Goodput by Policy across QPS')
    axes[0, 0].set_xlabel('Request Rate (QPS)')
    axes[0, 0].set_ylabel('Goodput')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(request_rates, rotation=45)

    # Subplot 2: SLO adherence by policy across QPS
    for i, policy in enumerate(policies):
        policy_data = per_qps_grouped[per_qps_grouped['policy'] == policy]
        values = [policy_data[policy_data['request_rate'] == rate]['slo_adherence'].mean() for rate in request_rates]
        axes[0, 1].bar(x + i * width - 0.4 + width/2, values, width, label=policy)
    axes[0, 1].set_title('SLO Adherence by Policy across QPS')
    axes[0, 1].set_xlabel('Request Rate (QPS)')
    axes[0, 1].set_ylabel('SLO Adherence')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(request_rates, rotation=45)

    # Legend for top subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Policy', bbox_to_anchor=(1.02, 0.5), loc='center left')

    # Subplot 3: Average goodput by policy
    avg_policies = sorted(avg_rows['policy'].unique())
    x2 = np.arange(len(avg_policies))
    values = [avg_rows[avg_rows['policy'] == p]['goodput'].mean() for p in avg_policies]
    axes[1, 0].bar(x2, values, 0.8)
    axes[1, 0].set_title('Average Goodput by Policy')
    axes[1, 0].set_xlabel('Policy')
    axes[1, 0].set_ylabel('Goodput')
    axes[1, 0].set_xticks(x2)
    axes[1, 0].set_xticklabels(avg_policies, rotation=45)

    # Subplot 4: Average SLO adherence by policy
    values = [avg_rows[avg_rows['policy'] == p]['slo_adherence'].mean() for p in avg_policies]
    axes[1, 1].bar(x2, values, 0.8)
    axes[1, 1].set_title('Average SLO Adherence by Policy')
    axes[1, 1].set_xlabel('Policy')
    axes[1, 1].set_ylabel('SLO Adherence')
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels(avg_policies, rotation=45)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'summary.png')
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
    
    # Calculate averages across runs for each policy (averaged across all request rates)
    avg_df = df.groupby(['policy']).agg({
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
