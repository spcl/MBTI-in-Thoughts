# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

PALETTE_BY_GROUP = {
    "INFJ ENTP ISFP": sns.color_palette("Blues", 1),
    "ESFJ INTP ENFJ": sns.color_palette("Greens", 1),
    "ISTP INFP ISFJ": sns.color_palette("Oranges", 1),
    "ESTP INTP ISFJ": sns.color_palette("Purples", 1),
    "NONE NONE NONE": sns.color_palette("Reds", 1),
    "EXPERT EXPERT EXPERT": sns.color_palette("Greys", 1),
}

PALETTE_BY_PROTOCOL = {
    "vote": sns.color_palette("Blues", 1),
}

def merge_benchmark_results(path: str) -> pd.DataFrame:
    all_dfs = []
    for root, _, files in os.walk(path):
        for filename in files:
            if not filename.endswith(".csv"):
                continue
            file_path = os.path.join(root, filename)
            parts = os.path.relpath(file_path, path).split(os.sep)
            dataset_owner, dataset, task, protocol, rest = parts[:5]
            run_no = int(rest.rsplit("_run_", 1)[1].split(".")[0])
            mbti_group = rest.split("_run_")[0]
            df = pd.read_csv(file_path)
            # Remove the first column if it is unnamed
            if df.columns[0].startswith("Unnamed:"):
                df = df.iloc[:, 1:]
            df["datasetowner"] = dataset_owner
            df["dataset"] = dataset
            df["task"] = task
            df["protocol"] = protocol
            df["MBTIGroup"] = mbti_group
            df["run"] = run_no
            df["score"] = (df["prediction"] == df["label"]).astype(int)
            # Move new columns to the front
            new_cols = ["datasetowner", "dataset", "task", "protocol", "MBTIGroup", "run"]
            df = df[new_cols + [c for c in df.columns if c not in new_cols]]
            all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        return merged_df
    else:
        print("No CSV files found to merge.")
        
def plot_average_bar_chart(
    df: pd.DataFrame,
    plts_path: str,
    x_col: str,
    y_col: str,
    color_col: str = None,
) -> None:
    """
    Plot a bar chart showing average scores grouped by x_col, optionally colored by color_col.
    """
    plt.figure()
    
    if color_col:
        grouped_df = df.groupby([x_col, color_col], as_index=False)[y_col].mean()
        sns.barplot(data=grouped_df, x=x_col, y=y_col, hue=color_col)
    else:
        grouped_df = df.groupby(x_col, as_index=False)[y_col].mean()
        sns.barplot(data=grouped_df, x=x_col, y=y_col, hue=x_col)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create filename
    color_part = color_col if color_col else "none"
    plt_path = os.path.join(plts_path, f"bar_{y_col}_{x_col}_{color_part}.png")
    
    plt.savefig(plt_path)
    plt.close()

if __name__ == "__main__":
    path = "data/results_"
    plts_path = "data/plots"
    os.makedirs(plts_path, exist_ok=True)
    df = merge_benchmark_results(path)
    # Set the style of seaborn
    sns.set(style="whitegrid")
    sns.set_palette("Set2")

    # Set the font size for all plots
    plt.rcParams.update({'font.size': 14})

    # Set the figure size for all plots
    plt.rcParams['figure.figsize'] = [10, 6]
    
    # plots dictionary x,y,hue
    plots_dict = {
        "bar":{
            "bar_score_protocol_protocol": {
                "x_col": "protocol",
                "y_col": "score"
            },
            "bar_score_dataset_protocol": {
                "x_col": "dataset",
                "y_col": "score",
                "color_col": "protocol"
            },
            "bar_score_task_protocol": {
                "x_col": "task",
                "y_col": "score",
                "color_col": "protocol"
            },
            "bar_score_mbti_protocol": {
                "x_col": "MBTIGroup",
                "y_col": "score",
                "color_col": "protocol"
            },
            "bar_score_mbti": {
                "x_col": "MBTIGroup",
                "y_col": "score"
            },
            "bar_score_mbti_task": {
                "x_col": "MBTIGroup",
                "y_col": "score",
                "color_col": "task"
            },
            "bar_score_task_mbti": {
                "x_col": "task",
                "y_col": "score",
                "color_col": "MBTIGroup"
            },
        }
    }
    # Loop through the plots dictionary and create the plots
    for plot_type, sub_plots in plots_dict.items():
        for plot_name, plot_info in sub_plots.items():
            x_col = plot_info["x_col"]
            y_col = plot_info["y_col"]
            color_col = plot_info.get("color_col", None)

            if plot_type == "bar":
                plot_average_bar_chart(
                    df,
                    plts_path,
                    x_col=x_col,
                    y_col=y_col,
                    color_col=color_col
                )
