#!/bin/bash

# Get the absolute path to the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARCHIVE_DIR="$SCRIPT_DIR/../archives"
RESULTS_DIR="$SCRIPT_DIR/../results"
COMBINED_RESULTS_FILE="$RESULTS_DIR/combined_results.csv"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Results directory $RESULTS_DIR does not exist. Exiting."
    exit 1
fi

# Check for existing results files
if [ -z "$(ls $RESULTS_DIR/results-viking-*.csv 2>/dev/null)" ]; then
    echo "No results files found in $RESULTS_DIR. Exiting."
    exit 1
fi

echo "Combining results..."

# Remove existing combined results file if it exists
if [ -f "$COMBINED_RESULTS_FILE" ]; then
    rm "$COMBINED_RESULTS_FILE"
fi

# Get the header from the first file and write it to the combined file
first_file=$(ls $RESULTS_DIR/results-viking-*.csv | head -n 1)
if [ -f "$first_file" ]; then
    head -n 1 "$first_file" > "$COMBINED_RESULTS_FILE"
fi

# Loop through all results files and append to the combined file (skipping the header)
for result_file in $RESULTS_DIR/results-viking-*.csv; do
    if [ -f "$result_file" ]; then
        echo "Adding $result_file to combined results..."
        tail -n +2 "$result_file" >> "$COMBINED_RESULTS_FILE"
        rm -f "$result_file"
    fi
done

echo "Results combined into $COMBINED_RESULTS_FILE"

# Archive returns files
echo "Archiving returns files ..."
bash "$SCRIPT_DIR/archive_returns.sh" "$ARCHIVE_DIR" "$RESULTS_DIR"
echo "All returns files are archived."
