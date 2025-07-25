#!/bin/bash
# Benchmark script for sequence length predictor

MODEL_CONFIG="model/results/opt-125m-llama3-8b-sharegpt-class-trainbucket82-b32/usage_config.json"
TEST_FILE="result/eval_results_opt-125m-llama3-8b-sharegpt-class-trainbucket82-b32_epoch_1.csv"
OUTPUT_DIR="benchmark_results"
LABEL_GROUP_SIZE=82
MIN_BATCH=1
MAX_BATCH=32
NUM_TRIALS=3
WARMUP_TRIALS=2
MAX_SAMPLES=1000

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the benchmark
python predictor/benchmark.py \
    --config $MODEL_CONFIG \
    --file $TEST_FILE \
    --output-dir $OUTPUT_DIR \
    --label-group-size $LABEL_GROUP_SIZE \
    --min-batch-size $MIN_BATCH \
    --max-batch-size $MAX_BATCH \
    --trials $NUM_TRIALS \
    --warmup-trials $WARMUP_TRIALS \
    --max-samples $MAX_SAMPLES

echo "Benchmark completed! Results saved to $OUTPUT_DIR"