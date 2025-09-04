# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

from collections import Counter
from evaluators import LexicalRichnessEvaluator, LMEvaluator, ReadabilityEvaluator


def load_evaluation_metrics(metrics_to_evaluate, evaluation_path):
    metrics = {metric: {} for metric in metrics_to_evaluate}
    evaluators = {
        metric: globals()[f"{metric}Evaluator"]()
        for metric in metrics_to_evaluate
    }

    for metric in metrics_to_evaluate:
        evaluator = evaluators[metric]
        evaluator_name = evaluator.__class__.__name__
        evaluator_dir = os.path.join(evaluation_path, evaluator_name)
        if not os.path.exists(evaluator_dir):
            continue

        for ptype in os.listdir(evaluator_dir):
            ptype_dir = os.path.join(evaluator_dir, ptype)
            if not os.path.isdir(ptype_dir):
                continue

            metric_names = evaluator.get_metric_names()
            if evaluator.is_numerical:
                metric_sums = {name: 0 for name in metric_names}
            else:
                metric_sums = {name: [] for name in metric_names}

            evaluation_files = os.listdir(ptype_dir)
            for eval_file in evaluation_files:
                file_path = os.path.join(ptype_dir, eval_file)
                with open(file_path, 'r') as f:
                    evaluation = json.load(f)
                for name in metric_names:
                    if evaluator.is_numerical:
                        metric_sums[name] += evaluation.get(name, 0)
                    else:
                        metric_sums[name].append(evaluation.get(name))

            if evaluator.is_numerical and evaluation_files:
                averages = {name: total / len(evaluation_files)
                            for name, total in metric_sums.items()}
                metrics[metric][ptype] = averages
            else:
                metrics[metric][ptype] = metric_sums

    return metrics, evaluators
    

def plot_metrics(metrics, evaluators, save_path):
    for metric_name, metric_contents in metrics.items():
        # Plotting the average metrics for each ptype
        metric_items = evaluators[metric_name].get_metric_names()

        for item in metric_items:
            fig, ax = plt.subplots(figsize=(10, 5))
            ptypes = list(metric_contents.keys())
            scores = [metrics[metric_name][ptype][item] for ptype in ptypes]
            
            ax.bar(ptypes, scores)
            ax.set_title(f'Average {item} Score by Personality Type')
            ax.set_xlabel('Personality Type')
            ax.set_ylabel(f'{item} Score')
            ax.set_xticklabels(ptypes, rotation=45, ha='right')
            
            # Save the plot for the current metric in the 'plots' directory
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plot_save_path = f'plots/{save_path}/{metric_name}/'
            # create save path if it does not exist
            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)
            fig.tight_layout()
            fig.savefig(f'{plot_save_path}{item}_score_plot.png')
            plt.close(fig)


def plot_llm_evaluations(metrics, evaluator, save_path):
    metric_items = evaluator.get_metric_names()
    # remove 'reasoning_for_evaluation' from the list of items to plot
    metric_items.remove('reasoning_for_evaluation')
    for item in metric_items:
        plot_metric_distribution(metrics['LLM'], item, save_path=save_path)


def save_llm_evaluatons_tabular(metrics, evaluator, save_path):
    metric_items = evaluator.get_metric_names()
    # remove 'reasoning_for_evaluation' from the list of items to plot
    metric_items.remove('reasoning_for_evaluation')
    for item in metric_items:
        save_metric_distribution_tabular(metrics['LLM'], item, save_path=save_path)  


def save_metric_distribution_tabular(evaluation_dict, metric, categories=None, use_percentages=True, save_path=None, file_formats=["xlsx", "pdf"]):
    """
    Saves the distribution of a given metric across MBTI types to specified file formats.
    Supported formats are Excel (.xlsx) and PDF.
    
    Rows represent MBTI types and columns represent score categories.
    The cell values are the percentage (or count, if use_percentages is False) of each category.
    
    :param evaluation_dict: Dict of the form:
        {
          'INTJ': {
            'positive_or_negative_story': [...],
            'happy_ending': [...],
            ...
          },
          'ENTJ': {
            'positive_or_negative_story': [...],
            ...
          },
          ...
        }
    :param metric: The key under each MBTI dictionary to process, e.g. 'positive_or_negative_story'.
    :param categories: List of possible categories for this metric, e.g. ['positive','negative','neutral'].
                       If None, categories are derived from the data (sorted for consistency).
    :param use_percentages: Whether to compute percentages. If False, raw counts are used.
    :param save_path: A subdirectory name where the files will be saved.
    :param file_formats: List of file formats to save the table. Supported: "xlsx", "pdf".
    """
    # 1. Gather data into a list of dictionaries: each representing (mbti, category, value).
    rows = []
    for ptype, metrics_dict in evaluation_dict.items():
        if metric not in metrics_dict:
            continue  # skip if this MBTI type doesn't have data for the metric
        
        answers = metrics_dict[metric]
        total = len(answers)
        counts = Counter(answers)
        
        # Determine which categories to include.
        cats = categories if categories else counts.keys()
        for cat in cats:
            count = counts[cat]
            if total > 0 and use_percentages:
                value = 100.0 * count / total
            else:
                value = count
            rows.append({"mbti": ptype, "category": str(cat), "value": value})
    
    # 2. If categories are not provided, derive them from the data.
    if categories is None:
        categories = sorted({row["category"] for row in rows})

    # 3. Build a DataFrame and pivot it so that rows are MBTI types and columns are categories.
    df = pd.DataFrame(rows)
    df_pivot = df.pivot(index="mbti", columns="category", values="value")
    df_pivot = df_pivot.reindex(columns=categories)
    df_pivot = df_pivot.fillna(0)  # Replace missing values with zero.
    
    # 4. Create a common directory for saving the files.
    base_save_path = f'tables/LLM/{save_path}/{metric}/'
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    
    # 5. Save as Excel if requested.
    if "xlsx" in file_formats:
        excel_file = f'{base_save_path}{metric}_score_table.xlsx'
        df_pivot.to_excel(excel_file)
    
    # 6. Save as PDF if requested.
    if "pdf" in file_formats:
        # Create a matplotlib figure with a table.
        # Format floating numbers to show only 4 leading decimals.
        formatted_values = [
            [f"{cell:.4f}" if isinstance(cell, float) else cell for cell in row]
            for row in df_pivot.values
        ]
        fig, ax = plt.subplots(figsize=(len(df_pivot.columns)*1.5, len(df_pivot)*0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=formatted_values,
                         colLabels=df_pivot.columns,
                         rowLabels=df_pivot.index,
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        fig.tight_layout()
        
        pdf_file = f'{base_save_path}{metric}_score_table.pdf'
        plt.savefig(pdf_file, bbox_inches='tight')
        plt.close(fig)


def plot_metric_distribution(evaluation_dict, metric, categories=None, use_percentages=True, save_path=None):
    """
    Plots the distribution of a given metric across MBTI types.
    
    :param evaluation_dict: Dict of the form:
        {
          'INTJ': {
            'positive_or_negative_story': [...],
            'happy_ending': [...],
            ...
          },
          'ENTJ': {
            'positive_or_negative_story': [...],
            ...
          },
          ...
        }
    :param metric: The key under each MBTI dictionary to plot, e.g. 'positive_or_negative_story'.
    :param categories: List of possible categories for this metric, e.g. ['positive','negative','neutral'].
                      If None, we derive categories from all answers found in the data (order will be sorted).
    :param use_percentages: Whether to convert counts to percentages in the stacked bar.
    """
    # 1. Gather data into a list of (mbti, category, value).
    rows = []
    for ptype, metrics_dict in evaluation_dict.items():
        if metric not in metrics_dict:
            continue  # skip if this MBTI type doesn't have data for the metric
        
        answers = metrics_dict[metric]
        total = len(answers)
        counts = Counter(answers)
        
        # If categories not given, we can derive from the unique answers.
        for cat in (categories if categories else counts.keys()):
            cat_count = counts[cat]
            if use_percentages:
                cat_value = 100.0 * cat_count / total if total > 0 else 0.0
            else:
                cat_value = cat_count
            rows.append({"mbti": ptype, "category": str(cat), "value": cat_value})
    
    # Derive categories if not provided; sort for consistency.
    if categories is None:
        categories = sorted({row["category"] for row in rows})

    # Build a color mapping using a discrete colormap (tab10) for distinct colors.
    cmap = plt.get_cmap("tab10")
    if len(categories) == 1:
        color_mapping = {categories[0]: cmap(0)}
    else:
        # Use modulo in case there are more than the available colors in tab10.
        color_mapping = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}
    
    # 2. Convert to DataFrame and pivot for stacked bar plotting.
    df = pd.DataFrame(rows)  
    df_pivot = df.pivot(index="mbti", columns="category", values="value")
    # Reindex to ensure column order matches the defined categories.
    df_pivot = df_pivot.reindex(columns=categories)
    
    # 3. Plot as a stacked bar with increased figure size and assign colors.
    fig, ax = plt.subplots(figsize=(14, 8))
    color_list = [color_mapping[cat] for cat in categories]
    df_pivot.plot(kind="bar", stacked=True, ax=ax, color=color_list)
    
    # 4. Add labels.
    metric_title = f"{metric} distribution"
    if use_percentages:
        metric_title += " (percent)"
    ax.set_title(metric_title)
    ax.set_ylabel("Percentage" if use_percentages else "Count")
    ax.set_xlabel("MBTI Type")
    
    # 5. Annotate each bar segment with its value.
    for container in ax.containers:
        for rect in container:
            height = rect.get_height()
            if height > 0:
                x = rect.get_x() + rect.get_width() / 2
                y = rect.get_y() + height / 2
                label = f"{height:.1f}%" if use_percentages else f"{height:.0f}"
                ax.text(x, y, label, ha='center', va='center', color='black', fontsize=8)
    
    # 6. Add legend & layout.
    plt.legend(title="Category")
    plt.tight_layout()
    
    final_save_path = f'plots/LLM/{save_path}/{metric}/'
    if not os.path.exists(final_save_path):
        os.makedirs(final_save_path)
    plt.savefig(f'{final_save_path}{metric}_score_plot.png', dpi=400)
    plt.close()


if __name__ == '__main__':
    metrics_to_evaluate = [
                            'Readability', 
                            # 'LexicalRichness', 
                            # 'LLM'
                            ]
    parser = argparse.ArgumentParser(description='Evaluate stories generated by the models primed with personality type descriptions')
    
    parser.add_argument('--plot_save_path', type=str, default='debugs', help='Where to save the plots')
    parser.add_argument('--table_save_path', type=str, default='debugs', help='Where to save the tables')
    parser.add_argument('--evaluation_path', type=str, default='evals_qwen3_readability', help='Where to save the evaluations')
    args = parser.parse_args()

    plot_save_path = args.plot_save_path
    table_save_path = args.table_save_path
    evaluation_path = ''.join(['evaluations/', args.evaluation_path])
    
    metrics, evaluators = load_evaluation_metrics(metrics_to_evaluate, evaluation_path)
    
    plot_metrics(metrics, evaluators, plot_save_path)
