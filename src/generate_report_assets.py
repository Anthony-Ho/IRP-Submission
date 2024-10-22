# Revised script with correct lowercase 'ewc' as per the user's data file

import os
import pandas as pd
import numpy as np
import argparse
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# Define functions
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / dof)
    return (x.mean() - y.mean()) / pooled_std

def average_cumulative_returns(data):
    table1 = data.groupby(['strategy', 'rl_model', 'data_group'])['Cumulative Return'].mean().unstack().round(4)
    return table1

def std_dev_cumulative_returns(data):
    table_std = data.groupby(['strategy', 'rl_model', 'data_group'])['Cumulative Return'].std().unstack().round(4)
    return table_std

def plasticity_analysis(data):
    group2_data = data[data['data_group'] == 'Group2']
    baseline = group2_data[group2_data['strategy'] == 'baseline']
    plasticity_results = []
    for strategy in ['naive', 'ewc', 'replay']:
        for model in ['PPO', 'A2C', 'DDPG']:
            strat_data = group2_data[(group2_data['strategy'] == strategy) & (group2_data['rl_model'] == model)]
            baseline_model = baseline[baseline['rl_model'] == model]
            t_stat, p_val = ttest_ind(baseline_model['Cumulative Return'], strat_data['Cumulative Return'], equal_var=False)
            effect_size = cohens_d(baseline_model['Cumulative Return'], strat_data['Cumulative Return'])
            plasticity_results.append({
                'rl_model': model,
                'strategy': f'baseline vs {strategy}',
                't_stat': round(t_stat, 4),
                'p_val': round(p_val, 4),
                'cohen_d': round(effect_size, 4)
            })
    plasticity_df = pd.DataFrame(plasticity_results)
    return plasticity_df

def stability_analysis(data):
    group1_data = data[data['data_group'] == 'Group1']
    baseline = group1_data[group1_data['strategy'] == 'baseline']
    stability_results = []
    for strategy in ['naive', 'ewc', 'replay']:
        for model in ['PPO', 'A2C', 'DDPG']:
            strat_data = group1_data[(group1_data['strategy'] == strategy) & (group1_data['rl_model'] == model)]
            baseline_model = baseline[baseline['rl_model'] == model]
            t_stat, p_val = ttest_ind(baseline_model['Cumulative Return'], strat_data['Cumulative Return'], equal_var=False)
            effect_size = cohens_d(baseline_model['Cumulative Return'], strat_data['Cumulative Return'])
            stability_results.append({
                'rl_model': model,
                'strategy': f'baseline vs {strategy}',
                't_stat': round(t_stat, 4),
                'p_val': round(p_val, 4),
                'cohen_d': round(effect_size, 4)
            })
    stability_df = pd.DataFrame(stability_results)
    return stability_df

def plot_boxplots(data, benchmark=0.0302, filename='cumulative_return_boxplots.png'):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    fig.suptitle('Box Plots of Cumulative Returns by Strategy and Data Group with DJIA Benchmark', fontsize=16)
    strategies = ['baseline', 'naive', 'ewc', 'replay']
    groups = ['Group1', 'Group2']
    
    for i, group in enumerate(groups):
        group_data = data[data['data_group'] == group]
        for j, strategy in enumerate(strategies):
            ax = axes[i, j]
            sns.boxplot(x='rl_model', y='Cumulative Return', data=group_data[group_data['strategy'] == strategy], ax=ax)
            ax.axhline(benchmark, color='red', linestyle='--', linewidth=1.2, label=f'DJIA Benchmark ({benchmark*100:.2f}%)')
            ax.set_title(f'{group} - {strategy.capitalize()}')
            ax.set_xlabel('')
            ax.set_ylabel('Cumulative Return' if j == 0 else '')
            # Add legend only to the first subplot
            if i == 0 and j == 0:
                ax.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.show()

# Main function with argument parsing
def main(input_file, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    data = pd.read_csv(input_file)

    # Generate tables and save to output directory
    table1 = average_cumulative_returns(data)
    table1.to_csv(f'{output_dir}/average_cumulative_returns.csv', index=True)

    table1_std = std_dev_cumulative_returns(data)
    table1_std.to_csv(f'{output_dir}/std_dev_cumulative_returns.csv', index=True)

    table2 = plasticity_analysis(data)
    table2.to_csv(f'{output_dir}/plasticity_analysis.csv', index=False)

    table3 = stability_analysis(data)
    table3.to_csv(f'{output_dir}/stability_analysis.csv', index=False)

    # Plot and save box plots
    plot_boxplots(data, 0.0302,  f'{output_dir}/cumulative_return_boxplots.png')

if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate tables and plots for cumulative return analysis.")
    parser.add_argument('--input_file', type=str, default='results/combined_results.csv', help="Path to the input CSV file.")
    parser.add_argument('--output_dir', type=str, default='report_assets', help="Directory to save output files.")
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(args.input_file, args.output_dir)
