#!/bin/bash

# Get the directory of the current script (so it works regardless of where it's run from)
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Define paths relative to the script directory
COMBINATIONS_RERUN="$SCRIPT_DIR/../data/combinations-rerun-experiments.csv"
COMBINATIONS_DEST="$SCRIPT_DIR/../data/combinations.csv"
MODELS_DIR="$SCRIPT_DIR/../models"
LOGS_DIR="$SCRIPT_DIR/../logs"
RESULTS_DIR="$SCRIPT_DIR/../results"

# Step 1: Copy combinations-rerun-experiments.csv to combinations.csv
if [ -f "$COMBINATIONS_RERUN" ]; then
    cp "$COMBINATIONS_RERUN" "$COMBINATIONS_DEST"
    echo "Copied $COMBINATIONS_RERUN to $COMBINATIONS_DEST"
else
    echo "Error: $COMBINATIONS_RERUN not found."
    exit 1
fi

# Step 2: Clean up the models directory
if [ -d "$MODELS_DIR" ]; then
    rm -rf "$MODELS_DIR"/*
    echo "Cleaned up $MODELS_DIR"
else
    echo "Error: $MODELS_DIR directory not found."
    exit 1
fi

# Step 3: Clean up the logs directory
if [ -d "$LOGS_DIR" ]; then
    rm -rf "$LOGS_DIR"/*
    echo "Cleaned up $LOGS_DIR"
else
    echo "Error: $LOGS_DIR directory not found."
    exit 1
fi

# Step 4: Remove combined_results.csv and any results*.csv or returns*.csv in the results directory
if [ -d "$RESULTS_DIR" ]; then
    rm -f "$RESULTS_DIR"/combined_results.csv
    rm -f "$RESULTS_DIR"/results*.csv
    rm -f "$RESULTS_DIR"/returns*.csv
    echo "Cleaned up $RESULTS_DIR: removed combined_results.csv, results*.csv, and returns*.csv"
else
    echo "Error: $RESULTS_DIR directory not found."
    exit 1
fi

echo "Environment reset completed. You are ready to re-run the experiment."
