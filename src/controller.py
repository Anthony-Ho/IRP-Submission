import os
import tarfile
import pandas as pd
import glob
import sys
import argparse

# Local Import
from experiment_config import tic_list, result_dir, model_dir
from run_experiment_with_validation import run_experiment_with_validation

archive_dir = '../archives'

# Ensure archive directory exists
if not os.path.exists(archive_dir):
    os.makedirs(archive_dir)

def archive_models(iteration):
    """
    Archive the models created in the current iteration into a tar file and clean up the models directory.
    """
    # Get the model files for this iteration
    model_files = glob.glob(os.path.join(model_dir, f'*_{iteration}.zip'))
    
    if not model_files:
        print(f"No models found for iteration {iteration}")
        return

    # Create the tar file for archiving
    tar_filename = os.path.join(archive_dir, f'models_iteration_{iteration}.tar.gz')
    with tarfile.open(tar_filename, 'w:gz') as tar:
        for model_file in model_files:
            tar.add(model_file, arcname=os.path.basename(model_file))
            os.remove(model_file)  # Remove the file after archiving

    print(f"Archived models for iteration {iteration} to {tar_filename}")

def run_controller(num_iterations, combination_file):
    """
    Main controller function to sequentially run experiments, archive models, and collect results.
    
    Parameters:
    - num_iterations: The number of iterations to run
    - combination_file: The path to the combination file for the experiment
    """
    for i in range(num_iterations):
        # Call run_experiment_with_validation() with the specified combination file
        iteration = run_experiment_with_validation(combination_file=combination_file)

        if iteration is None:
            print("No more combinations left to run.")
            break

        # Archive models for the completed iteration
        archive_models(iteration)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Controller script to run multiple experiment iterations.")
    parser.add_argument('num_iterations', type=int, help="Number of iterations to run.")
    parser.add_argument('--combination_file', type=str, default='combinations.csv', help="Path to the combination file.")

    # Parse arguments
    args = parser.parse_args()

    # Run controller with specified parameters
    run_controller(args.num_iterations, args.combination_file)