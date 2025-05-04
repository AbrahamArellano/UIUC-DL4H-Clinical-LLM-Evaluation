#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Results Analysis Script for "Do We Still Need Clinical Language Models?" Reproduction

This script analyzes the results from both fine-tuning and in-context learning experiments,
generates figures, and performs statistical analysis.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy.stats as stats

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUTPUT_DIR = "results/figures"
TASKS = ["mednli", "radqa", "clip"]
DATA_FRACTIONS = [1, 5, 10, 25, 100]
SHOTS = [1, 3, 5]
MODEL_PARAMS = {
    "t5-base": 220,
    "t5-large": 770,
    "roberta-large": 345,
    "clinical-t5-base": 220,
    "clinical-t5-large": 770,
    "BioClinRoBERTa": 345,
    "google/flan-t5-base": 250,
    "google/flan-t5-large": 770,
    "google/flan-t5-xl": 3000,
    "google/flan-t5-xxl": 11000,
    "gpt-3": 175000
}
MODEL_TYPES = {
    "t5-base": "general",
    "t5-large": "general",
    "roberta-large": "general",
    "clinical-t5-base": "clinical",
    "clinical-t5-large": "clinical",
    "BioClinRoBERTa": "clinical",
    "google/flan-t5-base": "general",
    "google/flan-t5-large": "general",
    "google/flan-t5-xl": "general",
    "google/flan-t5-xxl": "general",
    "gpt-3": "general"
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze results for clinical LLMs study.'
    )

    # Task selection
    parser.add_argument('--tasks', nargs='+', choices=['mednli', 'radqa', 'clip', 'all'], 
                        default=['all'], help='Tasks to analyze')

    # Directory paths
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save analysis results and figures')

    # Analysis options
    parser.add_argument('--skip_finetuning', action='store_true',
                        help='Skip fine-tuning analysis')
    parser.add_argument('--skip_icl', action='store_true',
                        help='Skip in-context learning analysis')
    parser.add_argument('--skip_comparison', action='store_true',
                        help='Skip comparison between fine-tuning and ICL')
    parser.add_argument('--skip_stats', action='store_true',
                        help='Skip statistical significance tests')

    # Figure options
    parser.add_argument('--figure_format', type=str, default='png',
                        help='Figure format (png, pdf, svg)')
    parser.add_argument('--figure_dpi', type=int, default=300,
                        help='Figure DPI')
    parser.add_argument('--figure_width', type=int, default=10,
                        help='Figure width in inches')
    parser.add_argument('--figure_height', type=int, default=6,
                        help='Figure height in inches')

    return parser.parse_args()


def load_finetuning_results(task: str, results_dir: str) -> pd.DataFrame:
    """
    Load fine-tuning results for a task.

    Args:
        task: Task name ('mednli', 'radqa', 'clip')
        results_dir: Directory containing results

    Returns:
        DataFrame of fine-tuning results
    """
    task_dir = os.path.join(results_dir, task)
    if not os.path.exists(task_dir):
        logger.warning(f"Task directory not found: {task_dir}")
        return pd.DataFrame()

    # Find all JSON result files
    result_files = []
    for root, _, files in os.walk(task_dir):
        for file in files:
            if file.endswith('.json') and 'stats' not in file:
                result_files.append(os.path.join(root, file))

    logger.info(f"Found {len(result_files)} result files for {task}")

    # Extract results from each file
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Skip ICL results
            if 'num_shots' in data:
                continue

            # Extract key information
            model = data.get('model', 'unknown')
            data_fraction = data.get('data_fraction', 0)

            # Convert string fractions to integers
            if isinstance(data_fraction, str) and data_fraction.endswith('pct'):
                data_fraction = int(data_fraction.rstrip('pct'))
            elif data_fraction == 'full':
                data_fraction = 100

            # Get metrics based on task
            metrics = {}
            if task == 'mednli':
                if 'test_metrics' in data:
                    metrics = data['test_metrics']
                else:
                    metrics = {k: v for k, v in data.items() 
                              if k in ['accuracy', 'f1', 'precision', 'recall']}
            elif task == 'radqa':
                if 'test_metrics' in data:
                    metrics = data['test_metrics']
                else:
                    metrics = {k: v for k, v in data.items() 
                              if k in ['exact_match', 'f1']}
            elif task == 'clip':
                if 'test_metrics' in data:
                    metrics = data['test_metrics']
                else:
                    metrics = {k: v for k, v in data.items() 
                              if k in ['micro_f1', 'macro_f1']}

            # Create result record
            result = {
                'task': task,
                'model': model,
                'data_fraction': data_fraction,
                'source': 'finetuned',
                **metrics
            }

            # Add model parameters
            model_key = model.split('/')[-1] if '/' in model else model
            result['params'] = MODEL_PARAMS.get(model_key, MODEL_PARAMS.get(model, 0))
            result['model_type'] = MODEL_TYPES.get(model_key, MODEL_TYPES.get(model, 'unknown'))

            results.append(result)
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    logger.info(f"Loaded {len(df)} fine-tuning results for {task}")

    return df


def load_icl_results(task: str, results_dir: str) -> pd.DataFrame:
    """
    Load in-context learning results for a task.

    Args:
        task: Task name ('mednli', 'radqa', 'clip')
        results_dir: Directory containing results

    Returns:
        DataFrame of ICL results
    """
    icl_dir = os.path.join(results_dir, "icl", f"{task}_results")
    complete_file = os.path.join(icl_dir, f"{task}_icl_complete_results.json")

    if not os.path.exists(complete_file):
        logger.warning(f"ICL results file not found: {complete_file}")

        # Try to find individual result files
        result_files = []
        for root, _, files in os.walk(icl_dir):
            for file in files:
                if file.endswith('.json') and 'complete' not in file:
                    result_files.append(os.path.join(root, file))

        logger.info(f"Found {len(result_files)} individual ICL result files for {task}")

        # Extract results from each file
        results = []
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Skip non-ICL results
                if 'num_shots' not in data:
                    continue

                # Extract key information
                model = data.get('model', 'unknown')
                shots = data.get('num_shots', 0)
                data_fraction = data.get('data_fraction', 0)

                # Get metrics based on task
                metrics = {}
                if task == 'mednli':
                    metrics = {k: v for k, v in data.items() 
                              if k in ['accuracy', 'f1', 'precision', 'recall']}
                elif task == 'radqa':
                    metrics = {k: v for k, v in data.items() 
                              if k in ['exact_match', 'f1']}
                elif task == 'clip':
                    metrics = {k: v for k, v in data.items() 
                              if k in ['micro_f1', 'macro_f1']}

                # Create result record
                result = {
                    'task': task,
                    'model': model,
                    'num_shots': shots,
                    'data_fraction': data_fraction,
                    'source': 'icl',
                    **metrics
                }

                # Add model parameters
                model_key = model.split('/')[-1] if '/' in model else model
                result['params'] = MODEL_PARAMS.get(model_key, MODEL_PARAMS.get(model, 0))
                result['model_type'] = MODEL_TYPES.get(model_key, MODEL_TYPES.get(model, 'unknown'))

                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(results)
        logger.info(f"Loaded {len(df)} ICL results for {task}")

        return df

    # Load from complete results file
    logger.info(f"Loading ICL results from {complete_file}")

    try:
        with open(complete_file, 'r') as f:
            data = json.load(f)

        results = []
        for model_name, df_map in data.items():
            for data_frac, shot_list in df_map.items():
                for run in shot_list:
                    if not run:  # Skip empty entries
                        continue

                    # Extract key information
                    model = run.get("model", model_name)
                    shots = run.get("num_shots", 0)
                    data_fraction = run.get("data_fraction", int(data_frac))

                    # Get metrics based on task
                    metrics = {}
                    if task == 'mednli':
                        metrics = {k: v for k, v in run.items() 
                                  if k in ['accuracy', 'f1', 'precision', 'recall']}
                    elif task == 'radqa':
                        metrics = {k: v for k, v in run.items() 
                                  if k in ['exact_match', 'f1']}
                    elif task == 'clip':
                        metrics = {k: v for k, v in run.items() 
                                  if k in ['micro_f1', 'macro_f1']}

                    # Create result record
                    result = {
                        'task': task,
                        'model': model,
                        'num_shots': shots,
                        'data_fraction': data_fraction,
                        'source': 'icl',
                        **metrics
                    }

                    # Add model parameters
                    model_key = model.split('/')[-1] if '/' in model else model
                    result['params'] = MODEL_PARAMS.get(model_key, MODEL_PARAMS.get(model, 0))
                    result['model_type'] = MODEL_TYPES.get(model_key, MODEL_TYPES.get(model, 'unknown'))

                    results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)
        logger.info(f"Loaded {len(df)} ICL results for {task}")

        return df

    except Exception as e:
        logger.error(f"Error loading ICL results from {complete_file}: {e}")
        return pd.DataFrame()


def analyze_finetuning_results(task: str, results_df: pd.DataFrame, output_dir: str, args):
    """
    Analyze fine-tuning results for a task.

    Args:
        task: Task name ('mednli', 'radqa', 'clip')
        results_df: DataFrame of fine-tuning results
        output_dir: Directory to save analysis results and figures
        args: Command-line arguments
    """
    # Create output directory
    task_output_dir = os.path.join(output_dir, task)
    os.makedirs(task_output_dir, exist_ok=True)

    if results_df.empty:
        logger.warning(f"No fine-tuning results to analyze for {task}")
        return

    logger.info(f"Analyzing fine-tuning results for {task}")

    # Determine main metric for task
    if task == 'mednli':
        main_metric = 'accuracy'
    elif task == 'radqa':
        main_metric = 'f1'
    elif task == 'clip':
        main_metric = 'macro_f1'
    else:
        main_metric = 'accuracy'

    # Create metric per parameter column
    results_df[f'{main_metric}_per_param'] = results_df[main_metric] / (results_df['params'] / 1e6)

    # 1. Create pivot table of results
    pivot_table = results_df.pivot_table(
        index=['model', 'model_type'],
        columns='data_fraction',
        values=main_metric,
        aggfunc='mean'
    ).round(3)

    # Save to CSV
    pivot_file = os.path.join(task_output_dir, f"{task}_finetuning_results.csv")
    pivot_table.to_csv(pivot_file)
    logger.info(f"Saved pivot table to {pivot_file}")

    # 2. Plot main metric vs. data fraction by model
    plt.figure(figsize=(args.figure_width, args.figure_height))

    for model_type in results_df['model_type'].unique():
        model_df = results_df[results_df['model_type'] == model_type]

        # Plot with error bars if we have multiple runs
        sns.lineplot(
            data=model_df,
            x='data_fraction',
            y=main_metric,
            label=model_type.capitalize(),
            marker='o' if model_type == 'clinical' else 's',
            ci=None
        )

    plt.title(f"{task.upper()}: {main_metric.capitalize()} vs. Training Data Fraction")
    plt.xlabel("Training Data Fraction (%)")
    plt.ylabel(main_metric.capitalize())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save figure
    fig_file = os.path.join(task_output_dir, f"{task}_metric_vs_fraction.{args.figure_format}")
    plt.savefig(fig_file, dpi=args.figure_dpi)
    plt.close()
    logger.info(f"Saved figure to {fig_file}")

    # 3. Plot efficiency (metric per parameter) by model
    plt.figure(figsize=(args.figure_width, args.figure_height))

    sns.barplot(
        data=results_df,
        x='model',
        y=f'{main_metric}_per_param',
        hue='model_type',
        palette={'clinical': 'orange', 'general': 'blue'},
        alpha=0.7
    )

    plt.title(f"{task.upper()}: Model Efficiency ({main_metric.capitalize()} per Million Parameters)")
    plt.xlabel("Model")
    plt.ylabel(f"{main_metric.capitalize()} per M Params")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save figure
    fig_file = os.path.join(task_output_dir, f"{task}_efficiency.{args.figure_format}")
    plt.savefig(fig_file, dpi=args.figure_dpi)
    plt.close()
    logger.info(f"Saved figure to {fig_file}")

    # 4. Compute statistics
    if not args.skip_stats:
        # Compute performance gap between general and clinical models
        gap_data = []

        # Group by data fraction and compute average performance by model type
        for frac in results_df['data_fraction'].unique():
            frac_df = results_df[results_df['data_fraction'] == frac]

            general_perf = frac_df[frac_df['model_type'] == 'general'][main_metric].mean()
            clinical_perf = frac_df[frac_df['model_type'] == 'clinical'][main_metric].mean()

            # Compute absolute and relative gaps
            abs_gap = clinical_perf - general_perf
            rel_gap = abs_gap / general_perf if general_perf > 0 else 0

            gap_data.append({
                'data_fraction': frac,
                'general_performance': general_perf,
                'clinical_performance': clinical_perf,
                'absolute_gap': abs_gap,
                'relative_gap': rel_gap
            })

        gap_df = pd.DataFrame(gap_data)

        # Save to CSV
        gap_file = os.path.join(task_output_dir, f"{task}_performance_gap.csv")
        gap_df.to_csv(gap_file, index=False)
        logger.info(f"Saved performance gap analysis to {gap_file}")

        # Plot performance gap
        plt.figure(figsize=(args.figure_width, args.figure_height))

        plt.bar(
            gap_df['data_fraction'],
            gap_df['absolute_gap'] * 100,  # Convert to percentage
            alpha=0.7
        )

        plt.title(f"{task.upper()}: Performance Gap (Clinical - General)")
                plt.xlabel("Training Data Fraction (%)")
                plt.ylabel("Performance Gap (%)")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Save figure
                fig_file = os.path.join(task_output_dir, f"{task}_performance_gap.{args.figure_format}")
                plt.savefig(fig_file, dpi=args.figure_dpi)
                plt.close()
                logger.info(f"Saved figure to {fig_file}")

                # Statistical significance testing
                # For each data fraction, test if clinical models significantly outperform general models
                for frac in results_df['data_fraction'].unique():
                    frac_df = results_df[results_df['data_fraction'] == frac]

                    clinical_scores = frac_df[frac_df['model_type'] == 'clinical'][main_metric].values
                    general_scores = frac_df[frac_df['model_type'] == 'general'][main_metric].values

                    if len(clinical_scores) > 0 and len(general_scores) > 0:
                        # Use Mann-Whitney U test (non-parametric)
                        try:
                            u_stat, p_value = stats.mannwhitneyu(clinical_scores, general_scores, alternative='two-sided')
                            logger.info(f"{task}, {frac}% data: Mann-Whitney U test p-value = {p_value:.4f}")

                            # Also compute mean and std
                            clinical_mean = clinical_scores.mean()
                            clinical_std = clinical_scores.std()
                            general_mean = general_scores.mean()
                            general_std = general_scores.std()

                            logger.info(f"  Clinical: {clinical_mean:.4f} ± {clinical_std:.4f}")
                            logger.info(f"  General: {general_mean:.4f} ± {general_std:.4f}")
                            logger.info(f"  Difference: {clinical_mean - general_mean:.4f}")
                        except Exception as e:
                            logger.warning(f"Statistical test failed for {task}, {frac}% data: {e}")
                    else:
                        logger.warning(f"Not enough data for statistical test: {task}, {frac}%")


        def analyze_icl_results(task: str, results_df: pd.DataFrame, output_dir: str, args):
            """
            Analyze in-context learning results for a task.

            Args:
                task: Task name ('mednli', 'radqa', 'clip')
                results_df: DataFrame of ICL results
                output_dir: Directory to save analysis results and figures
                args: Command-line arguments
            """
            # Create output directory
            task_output_dir = os.path.join(output_dir, task)
            os.makedirs(task_output_dir, exist_ok=True)

            if results_df.empty:
                logger.warning(f"No ICL results to analyze for {task}")
                return

            logger.info(f"Analyzing ICL results for {task}")

            # Determine main metric for task
            if task == 'mednli':
                main_metric = 'accuracy'
            elif task == 'radqa':
                main_metric = 'f1'
            elif task == 'clip':
                main_metric = 'macro_f1'
            else:
                main_metric = 'accuracy'

            # Create metric per parameter column
            results_df[f'{main_metric}_per_param'] = results_df[main_metric] / (results_df['params'] / 1e6)

            # 1. Create pivot table of results
            pivot_table = results_df.pivot_table(
                index=['model', 'model_type'],
                columns='num_shots',
                values=main_metric,
                aggfunc='mean'
            ).round(3)

            # Save to CSV
            pivot_file = os.path.join(task_output_dir, f"{task}_icl_results.csv")
            pivot_table.to_csv(pivot_file)
            logger.info(f"Saved pivot table to {pivot_file}")

            # 2. Plot main metric vs. number of shots by model
            plt.figure(figsize=(args.figure_width, args.figure_height))

            for model_type in results_df['model_type'].unique():
                model_df = results_df[results_df['model_type'] == model_type]

                # Plot with error bars if we have multiple runs
                sns.lineplot(
                    data=model_df,
                    x='num_shots',
                    y=main_metric,
                    label=model_type.capitalize(),
                    marker='o' if model_type == 'clinical' else 's',
                    ci=None
                )

            plt.title(f"{task.upper()}: {main_metric.capitalize()} vs. Number of Shots")
            plt.xlabel("Number of Few-Shot Examples")
            plt.ylabel(main_metric.capitalize())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save figure
            fig_file = os.path.join(task_output_dir, f"{task}_metric_vs_shots.{args.figure_format}")
            plt.savefig(fig_file, dpi=args.figure_dpi)
            plt.close()
            logger.info(f"Saved figure to {fig_file}")

            # 3. Plot efficiency (metric per parameter) by model
            plt.figure(figsize=(args.figure_width, args.figure_height))

            sns.barplot(
                data=results_df,
                x='model',
                y=f'{main_metric}_per_param',
                hue='model_type',
                palette={'clinical': 'orange', 'general': 'blue'},
                alpha=0.7
            )

            plt.title(f"{task.upper()} (ICL): Model Efficiency ({main_metric.capitalize()} per Million Parameters)")
            plt.xlabel("Model")
            plt.ylabel(f"{main_metric.capitalize()} per M Params")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save figure
            fig_file = os.path.join(task_output_dir, f"{task}_icl_efficiency.{args.figure_format}")
            plt.savefig(fig_file, dpi=args.figure_dpi)
            plt.close()
            logger.info(f"Saved figure to {fig_file}")

            # 4. Compute statistics
            if not args.skip_stats:
                # Compute performance gap between general and clinical models
                gap_data = []

                # Group by shot count and compute average performance by model type
                for shots in results_df['num_shots'].unique():
                    shots_df = results_df[results_df['num_shots'] == shots]

                    general_perf = shots_df[shots_df['model_type'] == 'general'][main_metric].mean()
                    clinical_perf = shots_df[shots_df['model_type'] == 'clinical'][main_metric].mean()

                    # Compute absolute and relative gaps
                    abs_gap = clinical_perf - general_perf
                    rel_gap = abs_gap / general_perf if general_perf > 0 else 0

                    gap_data.append({
                        'num_shots': shots,
                        'general_performance': general_perf,
                        'clinical_performance': clinical_perf,
                        'absolute_gap': abs_gap,
                        'relative_gap': rel_gap
                    })

                gap_df = pd.DataFrame(gap_data)

                # Save to CSV
                gap_file = os.path.join(task_output_dir, f"{task}_icl_performance_gap.csv")
                gap_df.to_csv(gap_file, index=False)
                logger.info(f"Saved ICL performance gap analysis to {gap_file}")

                # Plot performance gap
                plt.figure(figsize=(args.figure_width, args.figure_height))

                plt.bar(
                    gap_df['num_shots'],
                    gap_df['absolute_gap'] * 100,  # Convert to percentage
                    alpha=0.7
                )

                plt.title(f"{task.upper()} (ICL): Performance Gap (Clinical - General)")
                plt.xlabel("Number of Few-Shot Examples")
                plt.ylabel("Performance Gap (%)")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Save figure
                fig_file = os.path.join(task_output_dir, f"{task}_icl_performance_gap.{args.figure_format}")
                plt.savefig(fig_file, dpi=args.figure_dpi)
                plt.close()
                logger.info(f"Saved figure to {fig_file}")


        def compare_finetuning_vs_icl(task: str, ft_df: pd.DataFrame, icl_df: pd.DataFrame, output_dir: str, args):
            """
            Compare fine-tuning and ICL results for a task.

            Args:
                task: Task name ('mednli', 'radqa', 'clip')
                ft_df: DataFrame of fine-tuning results
                icl_df: DataFrame of ICL results
                output_dir: Directory to save analysis results and figures
                args: Command-line arguments
            """
            # Create output directory
            task_output_dir = os.path.join(output_dir, task)
            os.makedirs(task_output_dir, exist_ok=True)

            if ft_df.empty or icl_df.empty:
                logger.warning(f"Not enough data to compare fine-tuning and ICL for {task}")
                return

            logger.info(f"Comparing fine-tuning and ICL results for {task}")

            # Determine main metric for task
            if task == 'mednli':
                main_metric = 'accuracy'
            elif task == 'radqa':
                main_metric = 'f1'
            elif task == 'clip':
                main_metric = 'macro_f1'
            else:
                main_metric = 'accuracy'

            # Map ICL shots to equivalent data fractions for comparison
            shot_to_fraction = {
                1: 1,
                3: 5,
                5: 10
            }

            # Create a copy of ICL data with mapped data fractions
            icl_mapped = icl_df.copy()
            icl_mapped['data_fraction'] = icl_mapped['num_shots'].map(shot_to_fraction)

            # Filter to include only the mapped fractions
            icl_mapped = icl_mapped[icl_mapped['data_fraction'].isin(shot_to_fraction.values())]
            ft_mapped = ft_df[ft_df['data_fraction'].isin(shot_to_fraction.values())]

            # Combine the data
            icl_mapped['method'] = 'ICL'
            ft_mapped['method'] = 'Fine-tuning'
            combined = pd.concat([icl_mapped, ft_mapped])

            # 1. Create comparative plot (same model type, different methods)
            for model_type in ['clinical', 'general']:
                plt.figure(figsize=(args.figure_width, args.figure_height))

                model_data = combined[combined['model_type'] == model_type]

                sns.lineplot(
                    data=model_data,
                    x='data_fraction',
                    y=main_metric,
                    hue='method',
                    style='method',
                    markers=True,
                    dashes=True
                )

                plt.title(f"{task.upper()} - {model_type.capitalize()} Models: Fine-tuning vs. ICL")
                plt.xlabel("Training Data / Equivalent Shot Count")
                plt.ylabel(main_metric.capitalize())
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Save figure
                fig_file = os.path.join(task_output_dir, f"{task}_{model_type}_ft_vs_icl.{args.figure_format}")
                plt.savefig(fig_file, dpi=args.figure_dpi)
                plt.close()
                logger.info(f"Saved figure to {fig_file}")

            # 2. Compute relative efficiency (performance per parameter)
            # Create pivot table of efficiency
            ft_efficiency = ft_mapped.groupby(['model_type', 'data_fraction'])[f'{main_metric}_per_param'].mean().reset_index()
            icl_efficiency = icl_mapped.groupby(['model_type', 'data_fraction'])[f'{main_metric}_per_param'].mean().reset_index()

            ft_efficiency['method'] = 'Fine-tuning'
            icl_efficiency['method'] = 'ICL'

            efficiency_combined = pd.concat([ft_efficiency, icl_efficiency])

            # Plot efficiency comparison
            plt.figure(figsize=(args.figure_width, args.figure_height))

            sns.barplot(
                data=efficiency_combined,
                x='data_fraction',
                y=f'{main_metric}_per_param',
                hue='method',
                palette={'Fine-tuning': 'blue', 'ICL': 'green'},
                alpha=0.7
            )

            plt.title(f"{task.upper()}: Efficiency Comparison (Fine-tuning vs. ICL)")
            plt.xlabel("Training Data / Equivalent Shot Count")
            plt.ylabel(f"{main_metric.capitalize()} per M Params")
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save figure
            fig_file = os.path.join(task_output_dir, f"{task}_efficiency_comparison.{args.figure_format}")
            plt.savefig(fig_file, dpi=args.figure_dpi)
            plt.close()
            logger.info(f"Saved figure to {fig_file}")

            # 3. Create summary table
            summary_data = []

            for model_type in ['clinical', 'general']:
                for data_fraction in shot_to_fraction.values():
                    ft_perf = ft_mapped[(ft_mapped['model_type'] == model_type) & 
                                      (ft_mapped['data_fraction'] == data_fraction)][main_metric].mean()

                    icl_perf = icl_mapped[(icl_mapped['model_type'] == model_type) & 
                                        (icl_mapped['data_fraction'] == data_fraction)][main_metric].mean()

                    # Compute difference
                    diff = ft_perf - icl_perf
                    rel_diff = diff / icl_perf if icl_perf > 0 else 0

                    summary_data.append({
                        'task': task,
                        'model_type': model_type,
                        'data_fraction': data_fraction,
                        'fine_tuning_perf': ft_perf,
                        'icl_perf': icl_perf,
                        'absolute_diff': diff,
                        'relative_diff': rel_diff
                    })

            summary_df = pd.DataFrame(summary_data)

            # Save to CSV
            summary_file = os.path.join(task_output_dir, f"{task}_method_comparison.csv")
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Saved method comparison to {summary_file}")


        def prepare_cross_task_analysis(all_results: Dict[str, Dict[str, pd.DataFrame]], output_dir: str, args):
            """
            Prepare cross-task analysis combining results from all tasks.

            Args:
                all_results: Dictionary of results by task and method
                output_dir: Directory to save analysis results and figures
                args: Command-line arguments
            """
            cross_task_dir = os.path.join(output_dir, "cross_task")
            os.makedirs(cross_task_dir, exist_ok=True)

            logger.info("Preparing cross-task analysis")

            # Combine all fine-tuning results
            ft_dfs = []
            for task, results in all_results.items():
                if 'finetuned' in results and not results['finetuned'].empty:
                    ft_dfs.append(results['finetuned'])

            if not ft_dfs:
                logger.warning("No fine-tuning results to combine")
                return

            combined_ft = pd.concat(ft_dfs)

            # Normalize metrics for cross-task comparison
            metrics_by_task = {
                'mednli': 'accuracy',
                'radqa': 'f1',
                'clip': 'macro_f1'
            }

            # Create normalized metric column
            combined_ft['normalized_metric'] = float('nan')
            for task, metric in metrics_by_task.items():
                task_mask = combined_ft['task'] == task
                if metric in combined_ft.columns:
                    combined_ft.loc[task_mask, 'normalized_metric'] = combined_ft.loc[task_mask, metric]

            # Create metric per parameter column
            combined_ft['metric_per_param'] = combined_ft['normalized_metric'] / (combined_ft['params'] / 1e6)

            # 1. Plot performance by model type across tasks
            plt.figure(figsize=(args.figure_width, args.figure_height))

            sns.boxplot(
                data=combined_ft,
                x='task',
                y='normalized_metric',
                hue='model_type',
                palette={'clinical': 'orange', 'general': 'blue'},
                alpha=0.7
            )

            plt.title("Performance Comparison Across Tasks")
            plt.xlabel("Task")
            plt.ylabel("Performance Metric")
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save figure
            fig_file = os.path.join(cross_task_dir, f"cross_task_performance.{args.figure_format}")
            plt.savefig(fig_file, dpi=args.figure_dpi)
            plt.close()
            logger.info(f"Saved figure to {fig_file}")

            # 2. Plot performance gap by data fraction across tasks
            gap_data = []

            for task in combined_ft['task'].unique():
                for frac in combined_ft['data_fraction'].unique():
                    task_frac_df = combined_ft[(combined_ft['task'] == task) & (combined_ft['data_fraction'] == frac)]

                    general_perf = task_frac_df[task_frac_df['model_type'] == 'general']['normalized_metric'].mean()
                    clinical_perf = task_frac_df[task_frac_df['model_type'] == 'clinical']['normalized_metric'].mean()

                    # Compute gap
                    gap = clinical_perf - general_perf

                    gap_data.append({
                        'task': task,
                        'data_fraction': frac,
                        'gap': gap
                    })

            gap_df = pd.DataFrame(gap_data)

            # Plot the gap
            plt.figure(figsize=(args.figure_width, args.figure_height))

            sns.barplot(
                data=gap_df,
                x='data_fraction',
                y='gap',
                hue='task',
                alpha=0.7
            )

            plt.title("Performance Gap (Clinical - General) Across Tasks")
            plt.xlabel("Training Data Fraction (%)")
            plt.ylabel("Performance Gap")
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save figure
            fig_file = os.path.join(cross_task_dir, f"cross_task_performance_gap.{args.figure_format}")
            plt.savefig(fig_file, dpi=args.figure_dpi)
            plt.close()
            logger.info(f"Saved figure to {fig_file}")

            # 3. Create summary table for report
            summary_data = []

            for task in combined_ft['task'].unique():
                task_df = combined_ft[combined_ft['task'] == task]

                # Full data results
                full_data_df = task_df[task_df['data_fraction'] == 100]

                general_full = full_data_df[full_data_df['model_type'] == 'general']['normalized_metric'].mean()
                clinical_full = full_data_df[full_data_df['model_type'] == 'clinical']['normalized_metric'].mean()

                # 1% data results
                small_data_df = task_df[task_df['data_fraction'] == 1]

                general_small = small_data_df[small_data_df['model_type'] == 'general']['normalized_metric'].mean()
                clinical_small = small_data_df[small_data_df['model_type'] == 'clinical']['normalized_metric'].mean()

                # Compute relative performance changes with data reduction
                general_rel_change = (general_small - general_full) / general_full if general_full > 0 else 0
                clinical_rel_change = (clinical_small - clinical_full) / clinical_full if clinical_full > 0 else 0

                summary_data.append({
                    'task': task,
                    'general_100pct': general_full,
                    'clinical_100pct': clinical_full,
                    'general_1pct': general_small,
                    'clinical_1pct': clinical_small,
                    'general_rel_change': general_rel_change,
                    'clinical_rel_change': clinical_rel_change,
                    'gap_100pct': clinical_full - general_full,
                    'gap_1pct': clinical_small - general_small
                })

            summary_df = pd.DataFrame(summary_data)

            # Save to CSV
            summary_file = os.path.join(cross_task_dir, f"cross_task_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Saved cross-task summary to {summary_file}")

            # 4. Create a publication-ready table
            publication_table = summary_df[['task', 'general_100pct', 'clinical_100pct', 'gap_100pct',
                                         'general_1pct', 'clinical_1pct', 'gap_1pct']].round(3)

            # Save to CSV
            pub_table_file = os.path.join(cross_task_dir, f"publication_table.csv")
            publication_table.to_csv(pub_table_file, index=False)
            logger.info(f"Saved publication table to {pub_table_file}")


        def main():
            """Main entrypoint."""
            args = parse_args()

            logger.info("Starting results analysis")
            logger.info(f"Results directory: {args.results_dir}")
            logger.info(f"Output directory: {args.output_dir}")

            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)

            # Determine which tasks to analyze
            tasks_to_analyze = args.tasks
            if 'all' in tasks_to_analyze:
                tasks_to_analyze = TASKS

            # Track all results for cross-task analysis
            all_results = {}

            # Analyze each task
            for task in tasks_to_analyze:
                all_results[task] = {}

                # Load and analyze fine-tuning results
                if not args.skip_finetuning:
                    logger.info(f"Loading fine-tuning results for {task}")
                    ft_results = load_finetuning_results(task, args.results_dir)
                    all_results[task]['finetuned'] = ft_results

                    if not ft_results.empty:
                        analyze_finetuning_results(task, ft_results, args.output_dir, args)
                    else:
                        logger.warning(f"No fine-tuning results found for {task}")

                # Load and analyze ICL results
                if not args.skip_icl:
                    logger.info(f"Loading ICL results for {task}")
                    icl_results = load_icl_results(task, args.results_dir)
                    all_results[task]['icl'] = icl_results

                    if not icl_results.empty:
                        analyze_icl_results(task, icl_results, args.output_dir, args)
                    else:
                        logger.warning(f"No ICL results found for {task}")

                # Compare fine-tuning and ICL
                if not args.skip_comparison and not args.skip_finetuning and not args.skip_icl:
                    if task in all_results and 'finetuned' in all_results[task] and 'icl' in all_results[task]:
                        ft_df = all_results[task]['finetuned']
                        icl_df = all_results[task]['icl']

                        if not ft_df.empty and not icl_df.empty:
                            compare_finetuning_vs_icl(task, ft_df, icl_df, args.output_dir, args)

            # Cross-task analysis
            prepare_cross_task_analysis(all_results, args.output_dir, args)

            logger.info("Results analysis completed")


        if __name__ == "__main__":
            main()