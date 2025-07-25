import json
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.cm as cm
from scipy import stats as scipy_stats

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a piecewise model to predict prefill time.')
    parser.add_argument('--model_name', type=str, default='llama8b-sharegpt',
                        help='Model name for organizing results (e.g., llama8b-sharegpt-prefill)')
    parser.add_argument('--data_path', type=str, default='analytic_model/profile/llama8b-sharegpt-prefill.jsonl',
                        help='Path to the profiled prefill time data')
    parser.add_argument('--token_threshold', type=int, default=-5,
                        help='Token number threshold for piecewise model')
    return parser.parse_args()

def load_data(file_path):
    """Load data from prefill_time.jsonl file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_data(data):
    """Extract features and target from data."""
    token_nums = []
    step_times = []
    
    for item in data:
        token_nums.append(item['token_num'])
        # Convert step time from seconds to milliseconds
        step_times.append(item['step_time'] * 1000)  # s to ms conversion
    
    return np.array(token_nums), np.array(step_times)

def train_piecewise_model(token_nums, step_times, threshold):
    """Train a piecewise model: constant for small tokens, linear for large tokens."""
    # Split data into small and large token sets
    small_indices = token_nums <= threshold
    large_indices = token_nums > threshold
    
    small_tokens = token_nums[small_indices]
    small_times = step_times[small_indices]
    
    large_tokens = token_nums[large_indices]
    large_times = step_times[large_indices]
    
    # Calculate constant value for small tokens (mean of step_times)
    constant_value = np.mean(small_times)
    
    # Train linear model for large tokens
    linear_model = None
    alpha = 0
    beta = 0
    
    if len(large_tokens) > 0:
        X = large_tokens.reshape(-1, 1)
        linear_model = LinearRegression()
        linear_model.fit(X, large_times)
        
        # Extract coefficients
        alpha = linear_model.coef_[0]  # slope
        beta = linear_model.intercept_  # intercept
    
    return {
        'threshold': threshold,
        'constant_value': constant_value,
        'alpha': alpha,
        'beta': beta,
        'linear_model': linear_model
    }

def predict_piecewise(token_nums, model):
    """Make predictions using the piecewise model."""
    predictions = np.zeros_like(token_nums, dtype=float)
    
    # Apply constant value for small tokens
    small_indices = token_nums <= model['threshold']
    predictions[small_indices] = model['constant_value']
    
    # Apply linear model for large tokens
    large_indices = token_nums > model['threshold']
    if model['linear_model'] is not None and np.any(large_indices):
        large_tokens = token_nums[large_indices].reshape(-1, 1)
        predictions[large_indices] = model['linear_model'].predict(large_tokens)
    
    return predictions

def calculate_statistics(token_nums, y_true, y_pred):
    """Calculate comprehensive statistics for model evaluation."""
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate explained variance
    explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Calculate maximum error and median absolute error
    max_error = np.max(np.abs(residuals))
    median_abs_error = np.median(np.abs(residuals))
    
    # Calculate percentiles of absolute errors
    perc_90 = np.percentile(np.abs(residuals), 90)
    perc_95 = np.percentile(np.abs(residuals), 95)
    perc_99 = np.percentile(np.abs(residuals), 99)
    
    # Calculate F-statistic and p-value
    n = len(y_true)
    p = 1  # number of predictors (token_nums)
    rss = np.sum(residuals**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    f_statistic = ((tss - rss) / p) / (rss / (n - p - 1))
    p_value = 1 - scipy_stats.f.cdf(f_statistic, p, n-p-1)
    
    # Assemble statistics into a dictionary
    stats = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "explained_variance": explained_variance,
        "max_error": max_error,
        "median_abs_error": median_abs_error,
        "90th_percentile_error": perc_90,
        "95th_percentile_error": perc_95,
        "99th_percentile_error": perc_99,
        "f_statistic": f_statistic,
        "p_value": p_value
    }
    
    return stats, residuals

def print_statistics(stats):
    """Print comprehensive statistics for model evaluation."""
    print("\n======= PREFILL: MODEL EVALUATION STATISTICS =======")
    print(f"R² Score: {stats['r2']:.4f}")
    print(f"Explained Variance: {stats['explained_variance']:.4f}")
    print(f"Mean Squared Error (MSE): {stats['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {stats['rmse']:.4f}")
    print(f"Mean Absolute Error (MAE): {stats['mae']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {stats['mape']:.4f}%")
    print(f"Maximum Absolute Error: {stats['max_error']:.4f}")
    print(f"Median Absolute Error: {stats['median_abs_error']:.4f}")
    print(f"90th Percentile of Absolute Errors: {stats['90th_percentile_error']:.4f}")
    print(f"95th Percentile of Absolute Errors: {stats['95th_percentile_error']:.4f}")
    print(f"99th Percentile of Absolute Errors: {stats['99th_percentile_error']:.4f}")
    print("============================================")

def print_model_parameters(model):
    """Print the piecewise model parameters in a readable format."""
    print("\n======= PIECEWISE MODEL PARAMETERS =======")
    print(f"Token threshold: {model['threshold']}")
    print(f"Small tokens (≤ {model['threshold']}): constant value = {model['constant_value']:.4f} ms")
    print(f"Large tokens (> {model['threshold']}): prefill_time (ms) = {model['alpha']:.6f} * token_num + {model['beta']:.6f}")
    print("============================================")

def visualize_piecewise_model(token_nums, step_times, model, save_path):
    """Visualize the piecewise model with actual data points."""
    plt.figure(figsize=(12, 8))
    
    # Sort data for better visualization
    sorted_indices = np.argsort(token_nums)
    sorted_tokens = token_nums[sorted_indices]
    sorted_times = step_times[sorted_indices]
    
    # Generate prediction points for smoother line
    pred_tokens = np.linspace(min(token_nums), max(token_nums), 1000)
    
    # Make predictions using the piecewise model
    predictions = np.zeros_like(pred_tokens)
    small_indices = pred_tokens <= model['threshold']
    predictions[small_indices] = model['constant_value']
    
    if model['linear_model'] is not None:
        large_indices = pred_tokens > model['threshold']
        large_tokens = pred_tokens[large_indices].reshape(-1, 1)
        predictions[large_indices] = model['linear_model'].predict(large_tokens)
    
    # Scatter plot of actual data
    plt.scatter(sorted_tokens, sorted_times, alpha=0.6, label='Actual Data')
    
    # Plot the piecewise model
    plt.plot(pred_tokens, predictions, 'r-', linewidth=2, label='Piecewise Model')
    
    # Add vertical line at threshold
    plt.axvline(x=model['threshold'], color='g', linestyle='--', label=f'Threshold: {model["threshold"]} tokens')
    
    # Add labels and legend
    plt.xlabel('Token Number')
    plt.ylabel('Prefill Time (ms)')
    plt.title('Piecewise Prefill Time Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate R²
    actual_preds = predict_piecewise(token_nums, model)
    r2 = r2_score(step_times, actual_preds)
    
    # Add model equation text
    plt.text(0.02, 0.95, 
             f'Model:\n'
             f'  If tokens ≤ {model["threshold"]}: {model["constant_value"]:.2f} ms\n'
             f'  If tokens > {model["threshold"]}: {model["alpha"]:.6f} × tokens + {model["beta"]}:.2f ms\n'
             f'R² = {r2:.4f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_prediction_accuracy(y_true, y_pred, token_nums, save_path):
    """Visualize model prediction accuracy with actual vs predicted values."""
    plt.figure(figsize=(12, 10))
    
    # Main plot - True vs Predicted
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(y_true, y_pred, alpha=0.6, c=token_nums, cmap=cm.viridis)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)
    
    plt.xlabel('Actual Prefill Time (ms)')
    plt.ylabel('Predicted Prefill Time (ms)')
    plt.title('Actual vs Predicted Prefill Time')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Token Number')
    
    # Add R² text
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, c=token_nums, cmap=cm.viridis)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    
    plt.xlabel('Predicted Prefill Time (ms)')
    plt.ylabel('Residuals (ms)')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Token Number')
    
    # Add RMSE text
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.text(0.05, 0.95, f'RMSE = {rmse:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Error distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='-', linewidth=2)
    
    plt.xlabel('Error (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    
    # Calculate mean and std of residuals
    mean_error = np.mean(residuals)
    std_error = np.std(residuals)
    plt.text(0.05, 0.95, f'Mean = {mean_error:.4f}\nStd = {std_error:.4f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Percentage error plot
    plt.subplot(2, 2, 4)
    percentage_error = (y_true - y_pred) / y_true * 100
    scatter = plt.scatter(token_nums, percentage_error, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    
    plt.xlabel('Token Number')
    plt.ylabel('Percentage Error (%)')
    plt.title('Percentage Error vs Token Number')
    plt.grid(True, alpha=0.3)
    
    # Add MAPE text
    mape = np.mean(np.abs(percentage_error))
    plt.text(0.05, 0.95, f'MAPE = {mape:.4f}%', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def find_optimal_threshold(token_nums, step_times, min_threshold=50, max_threshold=1000, step=50):
    """Find the optimal threshold value by testing different thresholds."""
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    results = []
    
    for threshold in thresholds:
        model = train_piecewise_model(token_nums, step_times, threshold)
        y_pred = predict_piecewise(token_nums, model)
        r2 = r2_score(step_times, y_pred)
        rmse = np.sqrt(mean_squared_error(step_times, y_pred))
        
        results.append({
            'threshold': threshold,
            'r2': r2,
            'rmse': rmse,
            'model': model
        })
    
    # Sort by R² score (descending)
    sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)
    best_model = sorted_results[0]['model']
    
    # Visualize threshold optimization
    current_dir = Path(__file__).parent
    vis_dir = current_dir / "visualiztion" / "threshold_optimization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # R² plot
    plt.subplot(2, 1, 1)
    plt.plot(thresholds, [r['r2'] for r in results], 'b-o')
    plt.axvline(x=best_model['threshold'], color='r', linestyle='--', 
                label=f'Best Threshold: {best_model["threshold"]}')
    plt.xlabel('Threshold Value')
    plt.ylabel('R² Score')
    plt.title('R² Score vs Threshold Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # RMSE plot
    plt.subplot(2, 1, 2)
    plt.plot(thresholds, [r['rmse'] for r in results], 'g-o')
    plt.axvline(x=best_model['threshold'], color='r', linestyle='--', 
                label=f'Best Threshold: {best_model["threshold"]}')
    plt.xlabel('Threshold Value')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Threshold Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'threshold_optimization.png')
    plt.close()
    
    return best_model

def print_large_errors(token_nums, y_true, y_pred, threshold=1.0):
    """Print points where the prediction error is larger than threshold times."""
    errors = np.abs(y_true - y_pred)
    large_error_indices = errors > (threshold * np.minimum(y_true, y_pred))
    
    if np.any(large_error_indices):
        print(f"\n======= POINTS WITH >{threshold}x ERROR =======")
        print("Token Num | True Time (ms) | Pred Time (ms) | Error Ratio")
        print("--------------------------------------------------------")
        
        for idx in np.where(large_error_indices)[0]:
            error_ratio = errors[idx] / np.minimum(y_true[idx], y_pred[idx])
            print(f"{token_nums[idx]:8d} | {y_true[idx]:12.2f} | {y_pred[idx]:12.2f} | {error_ratio:10.2f}x")
        print("================================================")
    else:
        print(f"\nNo points found with error larger than {threshold}x")

def remove_outliers(token_nums, step_times, threshold=1.0):
    """Remove outliers based on prediction error."""
    # First fit a simple linear model to identify outliers
    X = token_nums.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, step_times)
    y_pred = model.predict(X)
    
    # Calculate errors and identify outliers
    errors = np.abs(step_times - y_pred)
    error_ratios = errors / np.minimum(step_times, y_pred)
    mask = error_ratios <= threshold
    
    return token_nums[mask], step_times[mask]

def compare_metrics(original_stats, filtered_stats):
    """Compare metrics between original and filtered data."""
    print("\n======= METRICS COMPARISON =======")
    print("Metric                | Original    | Filtered    | Improvement")
    print("------------------------------------------------------------")
    
    metrics = {
        'R² Score': ('r2', ''),
        'RMSE (ms)': ('rmse', '↓'),
        'MAE (ms)': ('mae', '↓'),
        'MAPE (%)': ('mape', '↓'),
        'Max Error (ms)': ('max_error', '↓'),
        'Median Error (ms)': ('median_abs_error', '↓')
    }
    
    for metric_name, (key, direction) in metrics.items():
        orig_val = original_stats[key]
        filt_val = filtered_stats[key]
        if direction == '↓':
            improvement = ((orig_val - filt_val) / orig_val) * 100
        else:
            improvement = ((filt_val - orig_val) / orig_val) * 100
        print(f"{metric_name:20} | {orig_val:10.4f} | {filt_val:10.4f} | {improvement:10.2f}%")
    print("=================================")

def main():
    # Parse command line arguments
    args = parse_arguments()
    model_name = args.model_name
    data_path = args.data_path
    token_threshold = args.token_threshold
    
    # Create output directories
    current_dir = Path(__file__).parent
    vis_dir = current_dir / "visualiztion" / model_name
    config_dir = current_dir / "config"
    
    # Create directories if they don't exist
    vis_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing model: {model_name}")
    print(f"Data path: {data_path}")
    print(f"Visualization directory: {vis_dir}")
    print(f"Config directory: {config_dir}")
    
    # Load data
    data = load_data(data_path)
    
    # Preprocess data
    token_nums, step_times = preprocess_data(data)
    
    # Train model on original data
    if token_threshold <= 0:
        print("Finding optimal threshold for original data...")
        original_model = find_optimal_threshold(token_nums, step_times)
        token_threshold = original_model['threshold']
        print(f"Optimal threshold found: {token_threshold}")
    else:
        original_model = train_piecewise_model(token_nums, step_times, token_threshold)
    
    # Calculate statistics for original data
    y_pred_original = predict_piecewise(token_nums, original_model)
    original_stats, _ = calculate_statistics(token_nums, step_times, y_pred_original)
    
    # Remove outliers
    print("\nRemoving outliers...")
    filtered_tokens, filtered_times = remove_outliers(token_nums, step_times)
    print(f"Removed {len(token_nums) - len(filtered_tokens)} outliers")
    
    # Train model on filtered data
    if token_threshold <= 0:
        print("Finding optimal threshold for filtered data...")
        filtered_model = find_optimal_threshold(filtered_tokens, filtered_times)
        filtered_threshold = filtered_model['threshold']
        print(f"Optimal threshold found: {filtered_threshold}")
    else:
        filtered_model = train_piecewise_model(filtered_tokens, filtered_times, token_threshold)
    
    # Calculate statistics for filtered data
    y_pred_filtered = predict_piecewise(filtered_tokens, filtered_model)
    filtered_stats, _ = calculate_statistics(filtered_tokens, filtered_times, y_pred_filtered)
    
    # Compare metrics
    compare_metrics(original_stats, filtered_stats)
    
    # Print model parameters
    print("\n======= ORIGINAL MODEL PARAMETERS =======")
    print_model_parameters(original_model)
    print("\n======= FILTERED MODEL PARAMETERS =======")
    print_model_parameters(filtered_model)
    
    # Save model coefficients to config.json
    config = {
        "model_name": model_name,
        "data_path": data_path,
        "model_coefficients": {
            "token_threshold": int(filtered_model['threshold']),
            "constant_value": float(filtered_model['constant_value']),
            "alpha": float(filtered_model['alpha']),
            "beta": float(filtered_model['beta'])
        },
        "model_statistics": filtered_stats,
        "units": "milliseconds"
    }
    
    # Save to model-specific config file
    config_path = config_dir / f"{model_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Generate visualizations
    visualize_piecewise_model(token_nums, step_times, original_model, 
                             vis_dir / 'original_piecewise_model.png')
    visualize_piecewise_model(filtered_tokens, filtered_times, filtered_model,
                             vis_dir / 'filtered_piecewise_model.png')
    
    visualize_prediction_accuracy(step_times, y_pred_original, token_nums,
                                vis_dir / 'original_prediction_accuracy.png')
    visualize_prediction_accuracy(filtered_times, y_pred_filtered, filtered_tokens,
                                vis_dir / 'filtered_prediction_accuracy.png')
    
    print(f"\nAnalysis complete. Results saved to:")
    print(f"- Config: {config_path}")
    print(f"- Visualizations: {vis_dir}")

if __name__ == "__main__":
    main()
