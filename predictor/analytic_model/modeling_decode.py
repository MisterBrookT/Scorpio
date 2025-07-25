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
from scipy import stats as scipy_stats

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a linear regression model to predict step time for decoding.')
    parser.add_argument('--model_name', type=str, default='llama8b-sharegpt',
                        help='Model name for organizing results (e.g., llama8b-sharegpt)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the profiled step time data')
    return parser.parse_args()

def load_data(file_path):
    """Load data from step_time.jsonl file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_data(data):
    """Extract features and target from data."""
    batch_sizes = []
    seq_lengths = []
    step_times = []
    
    for item in data:
        batch_sizes.append(item['batch_size'])
        seq_lengths.append(item['avg_seq_len'])
        # Convert step time from seconds to milliseconds
        step_times.append(item['step_time'] * 1000)  # s to ms conversion
    
    return np.array(batch_sizes), np.array(seq_lengths), np.array(step_times)

def create_features(batch_sizes, seq_lengths):
    """Create features for the model based on the form: a*batch_size*seq_length + b*batch_size + c*seq_length + d."""
    X = np.column_stack([
        batch_sizes * seq_lengths,  # Interaction term: batch_size * seq_length
        batch_sizes,                # batch_size term
        seq_lengths,                # seq_length term
        np.ones_like(batch_sizes)   # Intercept term
    ])
    return X

def train_model(X, y):
    """Train a linear regression model."""
    model = LinearRegression(fit_intercept=False)  # We already added intercept term
    model.fit(X, y)
    return model

def calculate_statistics(model, X, y):
    """Calculate comprehensive statistics for model evaluation."""
    # Basic predictions
    y_pred = model.predict(X)
    
    # Calculate various metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    # Calculate explained variance
    explained_variance = 1 - (np.var(y - y_pred) / np.var(y))
    
    # Calculate adjusted R²
    n = len(y)
    p = X.shape[1] - 1  # number of predictors (excluding intercept)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    
    # Cross-validation for robustness
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(LinearRegression(fit_intercept=False), X, y, cv=kf, scoring='r2')
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate maximum error and median absolute error
    max_error = np.max(np.abs(residuals))
    median_abs_error = np.median(np.abs(residuals))
    
    # Calculate percentiles of absolute errors
    perc_90 = np.percentile(np.abs(residuals), 90)
    perc_95 = np.percentile(np.abs(residuals), 95)
    perc_99 = np.percentile(np.abs(residuals), 99)
    
    # Calculate residual sum of squares (RSS) and total sum of squares (TSS)
    rss = np.sum(residuals**2)
    tss = np.sum((y - np.mean(y))**2)
    
    # F-statistic and p-value
    f_statistic = ((tss - rss) / p) / (rss / (n - p - 1))
    p_value = 1 - scipy_stats.f.cdf(f_statistic, p, n-p-1)
    
    # Assemble statistics into a dictionary
    stats = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mape": mape,
        "explained_variance": explained_variance,
        "cv_scores_mean": np.mean(cv_scores),
        "cv_scores_std": np.std(cv_scores),
        "max_error": max_error,
        "median_abs_error": median_abs_error,
        "90th_percentile_error": perc_90,
        "95th_percentile_error": perc_95,
        "99th_percentile_error": perc_99,
        "f_statistic": f_statistic,
        "p_value": p_value
    }
    
    return stats, y_pred, residuals

def print_statistics(stats):
    """Print comprehensive statistics for model evaluation."""
    print("\n======= DECODE: MODEL EVALUATION STATISTICS =======")
    print(f"R² Score: {stats['r2']:.4f}")
    print(f"Adjusted R² Score: {stats['adjusted_r2']:.4f}")
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
    print(f"Cross-Validation R² Scores (Mean ± Std): {stats['cv_scores_mean']:.4f} ± {stats['cv_scores_std']:.4f}")
    print(f"F-statistic: {stats['f_statistic']:.4f}")
    print(f"p-value: {stats['p_value']:.4f}")
    print("============================================")

def visualize_prediction_accuracy(y_true, y_pred, save_path):
    """Visualize model prediction accuracy with actual vs predicted values."""
    plt.figure(figsize=(12, 10))
    
    # Main plot - True vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)
    
    plt.xlabel('Actual Step Time (ms)')
    plt.ylabel('Predicted Step Time (ms)')
    plt.title('Actual vs Predicted Step Time')
    plt.grid(True, alpha=0.3)
    
    # Add R² text
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    
    plt.xlabel('Predicted Step Time (ms)')
    plt.ylabel('Residuals (ms)')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
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
    plt.scatter(y_true, percentage_error, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    
    plt.xlabel('Actual Step Time (ms)')
    plt.ylabel('Percentage Error (%)')
    plt.title('Percentage Error vs Actual Values')
    plt.grid(True, alpha=0.3)
    
    # Add MAPE text
    mape = np.mean(np.abs(percentage_error))
    plt.text(0.05, 0.95, f'MAPE = {mape:.4f}%', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_error_by_parameters(batch_sizes, seq_lengths, y_true, y_pred, save_path):
    """Visualize prediction errors across different parameter values."""
    plt.figure(figsize=(12, 10))
    
    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)
    relative_errors = abs_errors / y_true * 100
    
    # Errors vs batch size
    plt.subplot(2, 2, 1)
    plt.scatter(batch_sizes, abs_errors, alpha=0.6)
    plt.xlabel('Batch Size')
    plt.ylabel('Absolute Error (ms)')
    plt.title('Absolute Error vs Batch Size')
    plt.grid(True, alpha=0.3)
    
    # Fit a trend line
    if len(np.unique(batch_sizes)) > 1:
        z = np.polyfit(batch_sizes, abs_errors, 1)
        p = np.poly1d(z)
        plt.plot(np.unique(batch_sizes), p(np.unique(batch_sizes)), "r--", alpha=0.8)
    
    # Errors vs sequence length
    plt.subplot(2, 2, 2)
    plt.scatter(seq_lengths, abs_errors, alpha=0.6)
    plt.xlabel('Sequence Length')
    plt.ylabel('Absolute Error (ms)')
    plt.title('Absolute Error vs Sequence Length')
    plt.grid(True, alpha=0.3)
    
    # Fit a trend line
    z = np.polyfit(seq_lengths, abs_errors, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(seq_lengths), p(np.sort(seq_lengths)), "r--", alpha=0.8)
    
    # Relative errors vs batch size
    plt.subplot(2, 2, 3)
    plt.scatter(batch_sizes, relative_errors, alpha=0.6)
    plt.xlabel('Batch Size')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error vs Batch Size')
    plt.grid(True, alpha=0.3)
    
    # Fit a trend line
    if len(np.unique(batch_sizes)) > 1:
        z = np.polyfit(batch_sizes, relative_errors, 1)
        p = np.poly1d(z)
        plt.plot(np.unique(batch_sizes), p(np.unique(batch_sizes)), "r--", alpha=0.8)
    
    # Relative errors vs sequence length
    plt.subplot(2, 2, 4)
    plt.scatter(seq_lengths, relative_errors, alpha=0.6)
    plt.xlabel('Sequence Length')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error vs Sequence Length')
    plt.grid(True, alpha=0.3)
    
    # Fit a trend line
    z = np.polyfit(seq_lengths, relative_errors, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(seq_lengths), p(np.sort(seq_lengths)), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_prediction_bands(batch_sizes, seq_lengths, step_times, model, save_path):
    """Visualize prediction with confidence bands."""
    plt.figure(figsize=(10, 8))
    
    # Create a feature that combines batch size and sequence length
    combined_feature = batch_sizes * seq_lengths
    
    # Sort for better visualization
    sorted_indices = np.argsort(combined_feature)
    sorted_combined = combined_feature[sorted_indices]
    sorted_step_times = step_times[sorted_indices]
    
    # Make predictions
    X = create_features(batch_sizes, seq_lengths)
    y_pred = model.predict(X)
    sorted_predictions = y_pred[sorted_indices]
    
    # Calculate residuals and their standard deviation
    residuals = step_times - y_pred
    residual_std = np.std(residuals)
    
    # Plot the actual values
    plt.scatter(sorted_combined, sorted_step_times, alpha=0.6, label='Actual Values')
    
    # Plot the predictions
    plt.plot(sorted_combined, sorted_predictions, 'r-', linewidth=2, label='Model Predictions')
    
    # Add 95% prediction bands (±1.96 std)
    plt.fill_between(sorted_combined, 
                    sorted_predictions - 1.96 * residual_std,
                    sorted_predictions + 1.96 * residual_std,
                    alpha=0.2, color='blue', label='95% Prediction Band')
    
    plt.xlabel('Batch Size × Sequence Length')
    plt.ylabel('Step Time (ms)')
    plt.title('Step Time vs Batch Size × Sequence Length with 95% Prediction Bands')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² and RMSE text
    r2 = r2_score(step_times, y_pred)
    rmse = np.sqrt(mean_squared_error(step_times, y_pred))
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_robust_model_comparison(X, y, save_path):
    """Compare the performance of the linear model with more robust alternatives."""
    # Split data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(fit_intercept=False),
        'Huber Regression': None,  # Placeholder - we'll compute manually with scaled params
        'RANSAC Regression': None,  # Placeholder - we'll compute manually
    }
    
    # Try to import and use robust models if available
    try:
        from sklearn.linear_model import HuberRegressor, RANSACRegressor
        models['Huber Regression'] = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001)
        models['RANSAC Regression'] = RANSACRegressor(random_state=42)
    except ImportError:
        # Calculate predictions manually if imports fail
        pass
    
    # Fit models and calculate scores
    results = {}
    for name, model in models.items():
        if model is not None:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            results[name] = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'model': model
            }
    
    # Create comparison plot
    plt.figure(figsize=(10, 8))
    
    # Plot R² scores
    plt.subplot(2, 1, 1)
    names = list(results.keys())
    train_scores = [results[name]['train_r2'] for name in names]
    test_scores = [results[name]['test_r2'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Training R²')
    plt.bar(x + width/2, test_scores, width, label='Test R²')
    
    plt.ylabel('R² Score')
    plt.title('R² Score Comparison')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot RMSE scores
    plt.subplot(2, 1, 2)
    train_rmse = [results[name]['train_rmse'] for name in names]
    test_rmse = [results[name]['test_rmse'] for name in names]
    
    plt.bar(x - width/2, train_rmse, width, label='Training RMSE (ms)')
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE (ms)')
    
    plt.ylabel('RMSE (ms)')
    plt.title('RMSE Comparison')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return results

def print_model_coefficients(model):
    """Print the model coefficients in a readable format."""
    a, b, c, d = model.coef_
    print(f"Model Function: step_time (ms) = {a:.6f}*batch_size*seq_length + {b:.6f}*batch_size + {c:.6f}*seq_length + {d:.6f}")
    return a, b, c, d

def print_large_errors(batch_sizes, seq_lengths, y_true, y_pred, threshold=1.0):
    """Print points where the prediction error is larger than threshold times."""
    errors = np.abs(y_true - y_pred)
    large_error_indices = errors > (threshold * np.minimum(y_true, y_pred))
    
    if np.any(large_error_indices):
        print(f"\n======= POINTS WITH >{threshold}x ERROR =======")
        print("Batch Size | Seq Length | True Time (ms) | Pred Time (ms) | Error Ratio")
        print("----------------------------------------------------------------------")
        
        for idx in np.where(large_error_indices)[0]:
            error_ratio = errors[idx] / np.minimum(y_true[idx], y_pred[idx])
            print(f"{batch_sizes[idx]:10.1f} | {seq_lengths[idx]:10.1f} | {y_true[idx]:12.2f} | {y_pred[idx]:12.2f} | {error_ratio:10.2f}x")
        print("================================================")
    else:
        print(f"\nNo points found with error larger than {threshold}x")

def remove_outliers(batch_sizes, seq_lengths, step_times, threshold=1.0):
    """Remove outliers based on prediction error."""
    # First fit a simple linear model to identify outliers
    X = create_features(batch_sizes, seq_lengths)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, step_times)
    y_pred = model.predict(X)
    
    # Calculate errors and identify outliers
    errors = np.abs(step_times - y_pred)
    error_ratios = errors / np.minimum(step_times, y_pred)
    mask = error_ratios <= threshold
    
    return batch_sizes[mask], seq_lengths[mask], step_times[mask]

def compare_metrics(original_stats, filtered_stats):
    """Compare metrics between original and filtered data."""
    print("\n======= METRICS COMPARISON =======")
    print("Metric                | Original    | Filtered    | Improvement")
    print("------------------------------------------------------------")
    
    metrics = {
        'R² Score': ('r2', ''),
        'Adjusted R²': ('adjusted_r2', ''),
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
    batch_sizes, seq_lengths, step_times = preprocess_data(data)
    
    # Create features
    X = create_features(batch_sizes, seq_lengths)
    
    # Train model on original data
    original_model = train_model(X, step_times)
    
    # Calculate statistics for original data
    original_stats, y_pred_original, _ = calculate_statistics(original_model, X, step_times)
    
    # Remove outliers
    print("\nRemoving outliers...")
    filtered_batch_sizes, filtered_seq_lengths, filtered_step_times = remove_outliers(batch_sizes, seq_lengths, step_times)
    print(f"Removed {len(batch_sizes) - len(filtered_batch_sizes)} outliers")
    
    # Create features for filtered data
    filtered_X = create_features(filtered_batch_sizes, filtered_seq_lengths)
    
    # Train model on filtered data
    filtered_model = train_model(filtered_X, filtered_step_times)
    
    # Calculate statistics for filtered data
    filtered_stats, y_pred_filtered, _ = calculate_statistics(filtered_model, filtered_X, filtered_step_times)
    
    # Compare metrics
    compare_metrics(original_stats, filtered_stats)
    
    # Print model coefficients
    print("\n======= ORIGINAL MODEL COEFFICIENTS =======")
    a_orig, b_orig, c_orig, d_orig = print_model_coefficients(original_model)
    print("\n======= FILTERED MODEL COEFFICIENTS =======")
    a_filt, b_filt, c_filt, d_filt = print_model_coefficients(filtered_model)
    
    # Save model coefficients to config.json
    config = {
        "model_name": model_name,
        "data_path": data_path,
        "model_coefficients": {
            "a": float(a_filt),
            "b": float(b_filt),
            "c": float(c_filt),
            "d": float(d_filt)
        },
        "model_statistics": filtered_stats,
        "units": "milliseconds"
    }
    
    # Save to model-specific config file
    config_path = config_dir / f"{model_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Generate visualizations
    visualize_prediction_accuracy(step_times, y_pred_original, 
                                vis_dir / 'original_prediction_accuracy.png')
    visualize_prediction_accuracy(filtered_step_times, y_pred_filtered,
                                vis_dir / 'filtered_prediction_accuracy.png')
    
    visualize_error_by_parameters(batch_sizes, seq_lengths, step_times, y_pred_original,
                               vis_dir / 'original_error_by_parameters.png')
    visualize_error_by_parameters(filtered_batch_sizes, filtered_seq_lengths, filtered_step_times, y_pred_filtered,
                               vis_dir / 'filtered_error_by_parameters.png')
    
    visualize_prediction_bands(batch_sizes, seq_lengths, step_times, original_model,
                              vis_dir / 'original_prediction_bands.png')
    visualize_prediction_bands(filtered_batch_sizes, filtered_seq_lengths, filtered_step_times, filtered_model,
                              vis_dir / 'filtered_prediction_bands.png')
    
    print(f"\nAnalysis complete. Results saved to:")
    print(f"- Config: {config_path}")
    print(f"- Visualizations: {vis_dir}")

if __name__ == "__main__":
    main()
