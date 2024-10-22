#!/bin/bash

# Set default values
DEFAULT_COMBINATION_FILE="combinations.csv"
DEFAULT_ITERATIONS=1

# Check if the combination_file argument is provided, otherwise use default
COMBINATION_FILE="${1:-$DEFAULT_COMBINATION_FILE}"

# Check if the number of iterations is provided as the second argument, otherwise use default
ITERATIONS="${2:-$DEFAULT_ITERATIONS}"

# Define paths relative to the current directory
SRC_DIR="$PWD/src"
DATA_DIR="$PWD/data"
RESULTS_DIR="$PWD/results"
MODEL_DIR="$PWD/models"
ARCHIVE_DIR="$PWD/archives"

# Display which combination file and number of iterations are being used
echo "Using combination file: $COMBINATION_FILE"
echo "Running $ITERATIONS iterations"

# Loop through the specified number of iterations
for (( i=1; i<=ITERATIONS; i++ ))
do
    echo "Starting iteration $i..."

    docker run --gpus all -it \
    -v "$SRC_DIR:/workspace/IRP/src" \
    -v "$DATA_DIR:/workspace/IRP/data" \
    -v "$MODEL_DIR:/workspace/TRP/models" \
    -v "$RESULTS_DIR:/workspace/IRP/results" \
    rl-portfolio-optimization:latest --combination_file "$COMBINATION_FILE"

    echo "Completed iteration $i."
done

# Combine all results into a single CSV file
echo "Combining results using combine_results.sh..."
bash "$SRC_DIR/combine_results.sh"

echo "All iterations completed."