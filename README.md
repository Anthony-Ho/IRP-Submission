# Continual Learning in Reinforcement Learning-Based Portfolio Management: A Study on Adaptability and Knowledge Retention

## Overview

This project explores the application of Continual Learning (CL) techniques to Reinforcement Learning (RL) agents in the context of portfolio optimization. Using RL algorithms (PPO, A2C, and DDPG) and CL strategies (Naive, EWC, Replay Buffer), this research evaluates the agents' adaptability and stability across different stock groups. This artefact includes the source code, Docker configurations, and supporting resources for reproducing the experiment.

## Project Structure

```plaintext
.
├── archives/                    # Archived data and backups
├── data/                        # Directory for stock data and combinations files
│   ├── combinations-rerun-experiments.csv    # Only combinations for replicating the experiement
├── docker/                      # Docker configurations and Dockerfile
│   ├── Dockerfile
│   ├── requirements.txt
├── logs/                        # All logfiles
├── models/                      # Output directory for saved trained models
├── report_assets/               # Directory to store tables and charts for the report
│   ├── average_cumulative_returns.csv    # Table 1
│   ├── std_dev_cumulative_returns.csv    
│   ├── plasticity_analysis.csv           # Table 2
│   ├── stability_analysis.csv            # Table 3
│   └── cumulative_return_boxplots.png    # Figure 1
├── results/                     # Output directory for experiment results
│   ├── combined_results-with-early-stopping.csv    # The original test results of the experiments
├── src/                         # Source code for running experiments
│   ├── __init__.py
│   ├── combine_results.sh
│   ├── compare_results.py
│   ├── continual_learning.py
│   ├── controller.py
│   ├── data_processing.py
│   ├── envs.py
│   ├── experiment_config.py
│   ├── experiment_utils.py
│   ├── generate_report_assets.py
│   ├── performance.py
│   ├── requirements.txt
│   ├── reset_experiment.sh
│   ├── run_experiment_with_validation.py
│   ├── training.py
│   └── viking-crl.job
├── .gitignore
├── LICENSE
├── README.md                    # This README file
├── run_docker_experiment.sh     # Scripts to run the experiement in docker setup
└── Wai Cheong Ho Self-Assessment Ethics form.docx     # Approved Ethics Self-Assessment
```

## Prerequisites

- Docker with NVIDIA GPU support and drivers installed
- Python 3.9+ if running locally
- NVIDIA Docker Toolkit for GPU support in Docker

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Anthony-Ho/IRP.git
   cd IRP
   ```

2. **Build the Docker Image**: Navigate to the docker directory and build the Docker image:

    ```bash
    docker build -t rl-portfolio-optimization -f docker/Dockerfile .
    ```

3. **Prepare Data and Results Directories**: Ensure the data/ directory contains necessary data files, including the `combinations.csv` file for running the experiment.

4. **Run Experiments with Shell Script**: The project includes a shell script for running experiments with Docker. Use this to specify the combination file and the number of iterations.

    ```bash
    ./run_docker_experiment.sh [combination_file] [iterations]
    ```

    - `combination_file`: Optional. The filename to the combination file within `data/`. Defaults to `combinations.csv`.  The combination file must be located in `data/`.
    - `iterations`: Optional. Number of iterations to run. Defaults to 1.

    For example:

    ```bash
    ./run_docker_experiment.sh combinations-viking.csv 5
    ```

## Running Experiments

### Running a Single Experiment

You can run the experiment manually with Docker by using the `run_experiment_with_validation.py` script:

```bash
docker run --gpus all -it \
  -v "$PWD/src:/workspace/IRP/src" \
  -v "$PWD/data:/workspace/IRP/data" \
  -v "$PWD/results:/workspace/IRP/results" \
  rl-portfolio-optimization --combination_file "combinations-viking.csv"
```

### Running Multiple Iterations

The shell script `run_docker_experiment.sh` simplifies running multiple experiments in a loop. Adjust the number of iterations as needed.

## Collecting and Analyzing Results

After completing all experiment iterations, use the generate_report_assets.py script to aggregate and visualize performance metrics. This script will create summary tables and charts for easier analysis.

### Instructions

1. Run the Report Script:
    - Execute the following command to process results and generate the report assets:

        ```bash
        python src/generate_report_assets.py --input_file results/combined_results.csv --output_dir report_assets
        ```

    - The output, including summary tables and visualizations, will be saved in the report_assets/ directory.

2. Review Output Files:
    - Key files include:
      - `average_cumulative_returns.csv`: Average cumulative returns by strategy and model.
      - `plasticity_analysis.csv` and `stability_analysis.csv`: Comparative analyses of adaptability and retention.
      - `cumulative_return_boxplots.png`: Box plot of cumulative returns across strategies, with DJIA benchmark.

## Running Experiments in Viking Cluster

### Intructions

Follow the steps below to set up and run the experiment on the Viking HPC cluster:

1. Set Up the Environment

   - **Option 1**: Extract the artefact zip file to a directory of your choice.
   - **Option 2**: Clone the submission repository:

    ```bash
    git clone https://github.com/Anthony-Ho/IRP.git
    cd IRP/src
    ```

2. Edit the Job Script

    Before submitting the job, ensure the `viking-crl.job` file located in the `src/` directory is configured correctly. You will need to update the following:
    - **Job name**: Specify a meaningful job name for easy identification.
    - **Time limit**: Set the appropriate time limit based on your experiment's duration. Allocate approximately 70-100 minutes per iteration, including some buffer time, to ensure all iterations complete.
    - **Project Account**:  Update the project account field with a valid account for resource allocation.
    - **array**: Specify the number of parallel jobs to run using the `array` parameter.

3. Reset the Experiment Environment

    To reset and prepare the environment for a fresh run, execute the following script:

    ```bash
    ./reset_experiment.sh
    ```

    This will clean up previous experiment results, model files, and logs, and copy the required files for a new run.

4. Submit the Batch Job

    Submit the experiment as a batch job to the cluster:

    ```bash
    sbatch viking-crl.job
    ```

    This will start the specified number of parallel jobs according to the array settings in the job script.

5. Combine Results from Each Iteration

    After all iterations of the experiment have completed, combine the results by running the following command from the `src/` directory:

    ```bash
    ./combine_results.sh
    ```

    This will merge the individual iteration results into a single file for further analysis.

6. Collect and Analyze Results

    Once the results have been combined, use the following command to generate tables and visualizations for analysis. This command should also be executed from the `src/` directory:

    ``` bash
    python generate_report_assets.py --input_file ../results/combined_results.csv --output_dir ../report_assets
    ```

    The output will include summary tables and charts, which will be saved in the `report_assets/` directory for review.

## Files and Resources

- `requirements.txt`: List of dependencies for the Python environment.
- `run_experiment_with_validation.py`: Main experiment script, which trains and validates models using specified combination files.
- `data_processing.py`, `continual_learning.py`, `training.py`: Core modules for data preparation, continual learning strategies, and model training.
- `viking-crl.job`: Example job script for running the experiment on the Viking HPC cluster.

## Additional Notes

GPU Support: The Docker setup assumes NVIDIA GPU support. Ensure drivers and CUDA are correctly configured.
Output: Experiment results are saved in the `results/` directory. Performance metrics and returns are logged in CSV files for each iteration.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

This project builds upon the FinRL library and other related works in Reinforcement Learning and Financial Modeling.

This project utilizes and modifies code from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library. Specifically, the `train()` method in the `EWC_PPO`, `EWC_A2C`, and `EWC_DDPG` classes is based on the original `train()` implementation, with modifications to include the Elastic Weight Consolidation (EWC) penalty during loss calculation.
