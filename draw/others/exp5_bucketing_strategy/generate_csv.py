import os
import json
import pandas as pd
from pathlib import Path

def extract_metrics_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        if 'evaluate_metrics' in data:
            return data['evaluate_metrics']
    return None

def main():
    # Base directory containing the models
    base_dir = Path('predictor/seq_predictor/MODELS')
    
    # List to store all metrics
    all_metrics = []
    
    # Walk through all subdirectories
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Path to best_metrics.json
        metrics_file = model_dir / 'best_metrics.json'
        if not metrics_file.exists():
            continue
            
        # Extract metrics
        metrics = extract_metrics_from_json(metrics_file)
        if metrics:
            # Add model name as a column
            metrics['model_name'] = model_dir.name
            all_metrics.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns to put model_name first
    cols = ['model_name'] + [col for col in df.columns if col != 'model_name']
    df = df[cols]
    
    # Sort by model_name
    df = df.sort_values('model_name')
    
    # Format all float columns to 2 decimal places
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].round(3)
    
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    
    # Save to CSV in the same directory as the script
    output_path = script_dir / 'model_metrics.csv'
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    main()
