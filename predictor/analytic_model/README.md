# Analytic Model for Step Time Prediction

This module implements a linear regression model to predict step time based on batch size and sequence length.

## Model

The model uses the function form:

```
step_time (ms) = a*batch_size*sequence_length + b*batch_size + c*sequence_length + d
```

Where:
- `a`, `b`, `c`, and `d` are coefficients determined through linear regression
- `batch_size` is the number of samples processed in each step
- `sequence_length` is the average sequence length of the batch
- Step time is measured in milliseconds (ms)

## Statistical Evaluation

The model is evaluated using comprehensive statistical metrics:

- **R² Score**: 0.6558 - Indicates that about 65.6% of the variance in step time is explained by the model
- **Adjusted R² Score**: 0.6554 - R² adjusted for the number of predictors
- **Cross-Validation R² Score**: 0.8502 ± 0.2679 - Indicates model performance across different data subsets
- **Mean Squared Error (MSE)**: 33.4305 ms² - Average of squared errors
- **Root Mean Squared Error (RMSE)**: 5.7819 ms - Square root of MSE, in the same units as step time
- **Mean Absolute Error (MAE)**: 0.6250 ms - Average of absolute errors
- **Mean Absolute Percentage Error (MAPE)**: 2.6795% - Average percentage error
- **Maximum Absolute Error**: 280.1670 ms - Largest prediction error
- **Median Absolute Error**: 0.2019 ms - Median of absolute errors
- **F-statistic**: 1530.0266 - Measures of overall significance of the model

These metrics suggest the model has good predictive power for step time based on batch size and sequence length.

## Usage

Run the model training with the following command:

```bash
python analytic_model/modeling.py --model_name MODEL_NAME --data_path DATA_PATH
```

Arguments:
- `--model_name`: Name of the model for organizing results (e.g., "llama8b-sharegpt")
- `--data_path`: Path to the profiled step time data (default: "analytic_model/profile/step_time.jsonl")

For example:
```bash
python analytic_model/modeling.py --model_name llama8b-sharegpt --data_path analytic_model/profile/step_time.jsonl
```

This will:
1. Load data from the specified data path
2. Convert step times from seconds to milliseconds
3. Train a linear regression model
4. Output comprehensive model performance statistics
5. Save the model coefficients and statistics to `analytic_model/config/MODEL_NAME.json`
6. Generate visualizations in `analytic_model/visualiztion/MODEL_NAME/`

## Files

- `modeling.py`: Implementation of the linear regression model
- `config/MODEL_NAME.json`: Contains the trained model coefficients and statistics (in milliseconds)
- `profile/step_time.jsonl`: Input data with batch size, sequence length, and step time measurements (in seconds)

## Visualizations

The following visualizations are saved in the `analytic_model/visualiztion/MODEL_NAME/` directory:

1. **Prediction Accuracy** (`prediction_accuracy.png`)
   - Shows actual vs. predicted step times in milliseconds
   - Displays residuals and error distribution
   - Includes key metrics (R², RMSE, MAPE) for model evaluation

2. **Error by Parameters** (`error_by_parameters.png`)
   - Visualizes how errors vary across different batch sizes and sequence lengths
   - Shows both absolute errors (in ms) and relative errors with trend lines
   - Helps identify if certain parameter ranges have higher prediction errors

3. **Prediction Bands** (`prediction_bands.png`)
   - Displays the model predictions with 95% confidence bands in milliseconds
   - Shows the expected range of step times for given parameter values
   - Helps assess prediction reliability across the parameter space

4. **Model Comparison** (`model_comparison.png`)
   - Compares the performance of different regression models
   - Shows R² and RMSE (in ms) for training and test sets
   - Helps evaluate if more robust models might perform better
