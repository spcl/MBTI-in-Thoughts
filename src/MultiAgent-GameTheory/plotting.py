# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
from typing import Optional

df_types = {
    'pd':
        {
            'plots_path': 'data/outputs/plots/game_theory/stability_PD/',
            'df_path': 'data/outputs/stability_PD.csv',
        }
    ,
    'model':
        {
            'plots_path': 'data/outputs/plots/game_theory/by_model/',
            'df_path': 'data/outputs/by_model.csv',
            'to_keep': [
                        "deepseek-chat", 
                        "claude-3-haiku-20240307", 
                        "gemini-2.0-flash-lite",
                        "gpt-4o-mini-2024-07-18",
                        "gemini-2.5-pro-exp-03-25", 
                        "gpt-4.1-mini-2025-04-14", 
                        "gpt-3.5-turbo"
                    ]
        },
    'game':
        {
            'plots_path': 'data/outputs/plots/game_theory/by_game/',
            'df_path': 'data/outputs/by_game.csv',
            'to_keep': ['chicken', 'coordination', 'generic', 'hawk_dove', 'prisoners_dilemma', 'stag_hunt']
        }
}

personality_order = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ",
    "NONE", "SELFISH", "ALTRUISTIC"
]

MODEL_ALIAS = {
    "gemini-2.5-pro-exp-03-25":       "G2.5-pro",
    "gemini-2.0-flash-lite":          "G2.0-lite",
    "gpt-4o-mini-2024-07-18":         "GPT-4o-mini",
    "gpt-4.1-mini-2025-04-14":        "GPT-4.1-mini",
    "gpt-3.5-turbo":                  "GPT-3.5",
    "claude-3-haiku-20240307":        "Claude-3-hk",
    "deepseek-chat":                  "DeepSeek",
}

PALETTE_BY_PAIR = {
    "I/E": sns.color_palette("Blues",   2),
    "N/S": sns.color_palette("Greens",  2),
    "T/F": sns.color_palette("Oranges", 2),
    "J/P": sns.color_palette("Purples", 2),
}

HUE_COLOUR = {
    "I": PALETTE_BY_PAIR["I/E"][0], "E": PALETTE_BY_PAIR["I/E"][1],
    "N": PALETTE_BY_PAIR["N/S"][0], "S": PALETTE_BY_PAIR["N/S"][1],
    "T": PALETTE_BY_PAIR["T/F"][0], "F": PALETTE_BY_PAIR["T/F"][1],
    "J": PALETTE_BY_PAIR["J/P"][0], "P": PALETTE_BY_PAIR["J/P"][1],
}

PALETTE_BY_TYPE = {
    "INTJ": "#1f77b4", "INTP": "#ff7f0e", "ENTJ": "#2ca02c", "ENTP": "#d62728",
    "INFJ": "#9467bd", "INFP": "#8c564b", "ENFJ": "#e377c2", "ENFP": "#7f7f7f",
    "ISTJ": "#bcbd22", "ISFJ": "#17becf", "ESTJ": "#aec7e8", "ESFJ": "#ffbb78",
    "ISTP": "#98df8a", "ISFP": "#ff9896", "ESTP": "#c5b0d5", "ESFP": "#c49c94"
}

PALETTE_BY_TYPE.update({
    "NONE": (0.5, 0.5, 0.5),
    "ALTRUISTIC": (0.5, 0.5, 0.5),
    "SELFISH": (0.5, 0.5, 0.5)
})

LABEL_FULL = {
    "I": "Introversion", "E": "Extraversion",
    "N": "Intuition",    "S": "Sensing",
    "T": "Thinking",     "F": "Feeling",
    "J": "Judging",      "P": "Perceiving",
}

def who_lied_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns:
      * lied_first_1 — 1 if agent 1 lied first, 0 otherwise
      * lied_first_2 — 1 if agent 2 lied first, 0 otherwise
    Ties (‘they lied on the same turn’) count for both.
    If nobody ever lies, both columns are 0.
    """
    # Helper to locate first lie (0 → lie). Returns ∞ if no lies.
    def first_lie(seq):
        if isinstance(seq, list):
            try:
                return seq.index(0)
            except ValueError:
                pass          # 0 not in the list
        return np.inf

    # Find the first-lie index for each agent
    df = df.copy()   # Avoid mutating caller’s frame
    df["first_lie_1"] = df["truthful_agent_1"].apply(first_lie)
    df["first_lie_2"] = df["truthful_agent_2"].apply(first_lie)

    # Earliest lie (row-wise)
    earliest = df[["first_lie_1", "first_lie_2"]].min(axis=1)

    # Assign accountability; ensure at least one lie occurred
    df["lied_first_1"] = ((df["first_lie_1"] == earliest) & ~np.isinf(earliest)).astype(int)
    df["lied_first_2"] = ((df["first_lie_2"] == earliest) & ~np.isinf(earliest)).astype(int)

    # Optional: drop helper columns
    return df.drop(columns=["first_lie_1", "first_lie_2"])


def import_df_and_plot_path(df_type):
    df_path = df_types[df_type]['df_path']
    plots_path = df_types[df_type]['plots_path']
    to_keep = df_types[df_type].get('to_keep', None)
    df = pd.read_csv(df_path)
    df['agent_1_scores'] = df['agent_1_scores'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_2_scores'] = df['agent_2_scores'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_1_cumulative_scores'] = df['agent_1_scores'].apply(lambda x: [0] + [sum(x[:i+1]) for i in range(len(x))] if isinstance(x, list) else x)
    df['agent_2_cumulative_scores'] = df['agent_2_scores'].apply(lambda x: [0] + [sum(x[:i+1]) for i in range(len(x))] if isinstance(x, list) else x)

    # Parse as lists, if something went wrong when saving the csv
    df['truthful_agent_1'] = df['truthful_agent_1'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['truthful_agent_2'] = df['truthful_agent_2'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Parse the actions as lists of strings
    df['agent_1_actions'] = df['agent_1_actions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_2_actions'] = df['agent_2_actions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_1_messages'] = df['agent_1_messages'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['agent_2_messages'] = df['agent_2_messages'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['intent_agent_1'] = df['intent_agent_1'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['intent_agent_2'] = df['intent_agent_2'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    nonmbti = ['NONE', 'ALTRUISTIC', 'SELFISH']
    # Map each personality to its dichotomies
    def safe_dichotomy_extraction(x, position, positive_letter, negative_letter):
        if x in nonmbti:
            return None
        else:
            return positive_letter if x[position] == positive_letter else negative_letter

    nonmbti = ['NONE', 'ALTRUISTIC', 'SELFISH']
    # Map each personality to its dichotomies
    df['I/E_1'] = df['personality_1'].apply(lambda x: safe_dichotomy_extraction(x, 0, 'I', 'E'))
    df['N/S_1'] = df['personality_1'].apply(lambda x: safe_dichotomy_extraction(x, 1, 'N', 'S'))
    df['T/F_1'] = df['personality_1'].apply(lambda x: safe_dichotomy_extraction(x, 2, 'F', 'T'))
    df['J/P_1'] = df['personality_1'].apply(lambda x: safe_dichotomy_extraction(x, 3, 'J', 'P'))
    df['I/E_2'] = df['personality_2'].apply(lambda x: safe_dichotomy_extraction(x, 0, 'I', 'E'))
    df['N/S_2'] = df['personality_2'].apply(lambda x: safe_dichotomy_extraction(x, 1, 'N', 'S'))
    df['T/F_2'] = df['personality_2'].apply(lambda x: safe_dichotomy_extraction(x, 2, 'F', 'T'))
    df['J/P_2'] = df['personality_2'].apply(lambda x: safe_dichotomy_extraction(x, 3, 'J', 'P'))
    # Filter the DataFrame to keep only the specified games or model depending on the df_type
    if to_keep is not None:
        if df_type == 'model':
            df = df[df['model_name_1'].isin(to_keep)]
        elif df_type == 'game':
            df = df[df['game_name'].isin(to_keep)]
            
    df = who_lied_first(df)
    
    return df, plots_path


def import_df_agents(df):
    def count_strategy_switches(actions):
        if not actions or len(actions) < 2:
            return 0
        return sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
    # Create a new DataFrame with one row per agent
    df_agent1 = df.copy()
    df_agent2 = df.copy()

    # Add a column to indicate the agent
    df_agent1['Agent'] = 'Agent 1'
    df_agent2['Agent'] = 'Agent 2'
    # Rename columns to remove agent-specific prefixes
    df_agent1 = df_agent1.rename(columns={
        'personality_1': 'Personality',
        'agent_1_scores': 'Scores',
        'agent_1_messages': 'Messages',
        'agent_1_actions': 'Actions',
        'agent_1_cumulative_scores': 'CumulativeScore',
        'intent_agent_1': 'Intent',
        'truthful_agent_1': 'Truthful',
        'analysis_agent_1': 'Analysis',
        'lied_first_1': 'LiedFirst',
        'I/E_1': 'I/E',
        'N/S_1': 'N/S',
        'T/F_1': 'T/F',
        'J/P_1': 'J/P'
    })

    df_agent2 = df_agent2.rename(columns={
        'personality_2': 'Personality',
        'agent_2_scores': 'Scores',
        'agent_2_messages': 'Messages',
        'agent_2_actions': 'Actions',
        'agent_2_cumulative_scores': 'CumulativeScore',
        'intent_agent_2': 'Intent',
        'truthful_agent_2': 'Truthful',
        'analysis_agent_2': 'Analysis',
        'lied_first_2': 'LiedFirst',
        'I/E_2': 'I/E',
        'N/S_2': 'N/S',
        'T/F_2': 'T/F',
        'J/P_2': 'J/P'
    })

    # Remove the other columns
    df_agent1 = df_agent1.drop(columns=[
        'personality_2', 'agent_2_scores', 'agent_2_messages', 'agent_2_actions', 
        'agent_2_cumulative_scores', 'intent_agent_2', 'truthful_agent_2', 'analysis_agent_2',
        'I/E_2', 'N/S_2', 'T/F_2', 'J/P_2','lied_first_2'
    ])
    df_agent2 = df_agent2.drop(columns=[
        'personality_1', 'agent_1_scores', 'agent_1_messages', 'agent_1_actions', 
        'agent_1_cumulative_scores', 'intent_agent_1', 'truthful_agent_1', 'analysis_agent_1',
        'I/E_1', 'N/S_1', 'T/F_1', 'J/P_1', 'lied_first_1'
    ])
    # Combine the two DataFrames
    df_agents = pd.concat([df_agent1, df_agent2], ignore_index=True)
    
    # Small fixes
    df_agents = df_agents.rename(columns={'game_name': 'GameName','model_name_1': 'Model'})
    df_agents["TotalScore"] = df_agents["CumulativeScore"].apply(lambda x: x[-1])
    df_agents["Truthfulness"] = df_agents["Truthful"].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)
    df_agents['DefectionRatio'] = df_agents.apply(
        lambda row: (
            None
            if not row['Actions']
               or any(a not in ('defect', 'cooperate') for a in row['Actions'])
            else row['Actions'].count('defect') / len(row['Actions'])
        ),
        axis=1
    )
    df_agents['Switches'] = df_agents['Actions'].apply(count_strategy_switches)
        
    return df_agents

# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------

def _get_slice_col(df_type: str | None) -> str | None:
    if df_type == "game":
        return "GameName"
    if df_type == "model":
        return "Model"
    return None # pooled


def get_defection_rates_dichos(
    df: pd.DataFrame,
    plots_path: str | os.PathLike,
    df_type: str | None = None,
):
    import ast, numpy as np, os
    from scipy import stats

    slice_by = _get_slice_col(df_type)
    label    = slice_by if slice_by else "pooled"

    def defect_rate(actions):
        if isinstance(actions, str):
            actions = ast.literal_eval(actions)
        return np.mean([str(a).strip().lower() == "defect" for a in actions])

    df = df.copy()
    df["defect_rate"] = df["Actions"].apply(defect_rate)

    # Collapse to one row per personality (+ slice)
    g_cols = ["Personality", "I/E", "N/S", "T/F", "J/P"]
    if slice_by:
        g_cols.insert(0, slice_by)

    means = (
        df.groupby(g_cols, as_index=False)["defect_rate"]
          .mean()
          .dropna(subset=["defect_rate"])
    )

    # Welch t‑tests
    axes = {"I/E": ("I", "E"), "N/S": ("N", "S"), "T/F": ("T", "F"), "J/P": ("J", "P")}
    lines = []
    for dim, (a_lbl, b_lbl) in axes.items():
        a = means.loc[means[dim] == a_lbl, "defect_rate"].values
        b = means.loc[means[dim] == b_lbl, "defect_rate"].values
        if min(len(a), len(b)) < 2:
            lines.append(f"{dim}: not enough data (n_{a_lbl}={len(a)}, n_{b_lbl}={len(b)})")
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt(((a.var(ddof=1)*(len(a)-1)) + (b.var(ddof=1)*(len(b)-1)))
                            / (len(a)+len(b)-2))
        d = (b.mean() - a.mean()) / pooled_sd
        lines.append(
            f"{dim}: Mean {a_lbl}={a.mean():.3f}, {b_lbl}={b.mean():.3f} "
            f"(p={p:.3g}, d={d:.2f}, n={len(a)} vs {len(b)})"
        )

    os.makedirs(plots_path, exist_ok=True)
    with open(os.path.join(plots_path, f"{label}_defection_dichos.txt"), "w") as f:
        f.write("\n".join(lines))


def get_truthfulness_dichos(
    df: pd.DataFrame,
    plots_path: str | os.PathLike,
    df_type: str | None = None,
):
    import numpy as np, os
    from scipy import stats

    slice_by = _get_slice_col(df_type)
    label    = slice_by if slice_by else "pooled"

    g_cols = ["Personality", "I/E", "N/S", "T/F", "J/P"]
    if slice_by:
        g_cols.insert(0, slice_by)

    means = (
        df.groupby(g_cols, as_index=False)["Truthfulness"]
          .mean()
          .dropna(subset=["Truthfulness"])
    )

    axes = {"I/E": ("I", "E"), "N/S": ("N", "S"), "T/F": ("T", "F"), "J/P": ("J", "P")}
    lines = []
    for dim, (a_lbl, b_lbl) in axes.items():
        a = means.loc[means[dim] == a_lbl, "Truthfulness"].values
        b = means.loc[means[dim] == b_lbl, "Truthfulness"].values
        if min(len(a), len(b)) < 2:
            lines.append(f"{dim}: not enough data (n_{a_lbl}={len(a)}, n_{b_lbl}={len(b)})")
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt(((a.var(ddof=1)*(len(a)-1)) + (b.var(ddof=1)*(len(b)-1)))
                            / (len(a)+len(b)-2))
        d = (b.mean() - a.mean()) / pooled_sd
        lines.append(
            f"{dim}: Mean {a_lbl}={a.mean():.3f}, {b_lbl}={b.mean():.3f} "
            f"(p={p:.3g}, d={d:.2f}, n={len(a)} vs {len(b)})"
        )

    os.makedirs(plots_path, exist_ok=True)
    with open(os.path.join(plots_path, f"{label}_truthfulness_dichos.txt"), "w") as f:
        f.write("\n".join(lines))


# ── strategy‑switch dichotomies (concise report) ---------------------------
def get_strategy_switch_pvalues(
    df: pd.DataFrame,
    plots_path: str | os.PathLike,
    df_type: str | None = None,
):
    slice_by = _get_slice_col(df_type)
    label    = slice_by if slice_by else "Overall"

    df = df.rename(columns={"StrategySwitches": "Switches"}).copy()

    # One row = personality (+ slice) mean switches across 7 rounds
    g_cols = ["Personality", "I/E", "N/S", "T/F", "J/P"]
    if slice_by:
        g_cols.insert(0, slice_by)

    means = (
        df.groupby(g_cols, as_index=False)["Switches"]
          .mean()
          .dropna(subset=["Switches"])
    )

    axes  = {"I/E": ("I", "E"),
             "N/S": ("N", "S"),
             "T/F": ("T", "F"),
             "J/P": ("J", "P")}
    lines = []

    for dim, (a_lbl, b_lbl) in axes.items():
        a = means.loc[means[dim] == a_lbl, "Switches"].values
        b = means.loc[means[dim] == b_lbl, "Switches"].values
        n_a, n_b = len(a), len(b)

        if min(n_a, n_b) < 2:
            lines.append(f"{dim}: not enough data (n_{a_lbl}={n_a}, n_{b_lbl}={n_b})")
            continue

        p = ttest_ind(a, b, equal_var=False).pvalue # Welch p‑value
        higher = a_lbl if a.mean() > b.mean() else b_lbl

        lines.append(
            f"{dim}: n_{a_lbl}={n_a}, n_{b_lbl}={n_b}  "
            f"{a_lbl}={a.mean():.2f}, {b_lbl}={b.mean():.2f}  "
            f"(Welch p={p:.3g}) — {higher} more switches"
        )

    os.makedirs(plots_path, exist_ok=True)
    fname = os.path.join(plots_path, f"strategy_switch_pvalues_{label}.txt")
    with open(fname, "w") as f:
        f.write("\n".join(lines))

    return lines


# ---------- helper for proportion p-value -----------------------------------
def _prop_test(x1: int, n1: int, x2: int, n2: int) -> float:
    """
    Welch χ² two-proportion test; falls back to Fisher if cells < 5.
    """
    tbl = np.array([[x1, n1 - x1],
                    [x2, n2 - x2]])
    if (tbl < 5).any():
        _, p = fisher_exact(tbl, alternative="two-sided")
    else:
        _, p, _, _ = chi2_contingency(tbl, correction=False)
    return p


def _prop_test(x1, n1, x2, n2):
    tab = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    if (tab < 5).any():
        return fisher_exact(tab)[1]
    return chi2_contingency(tab, correction=False)[1]


def get_liedfirst_pvalues(
    df: pd.DataFrame,
    plots_path: str | os.PathLike,
    df_type: Optional[str] = None,
):
    # 0. Slice column
    slice_by = {"game": "GameName", "model": "Model"}.get(df_type)
    label    = slice_by if slice_by else "Overall"

    df        = df.copy()
    df["LiedFirst"] = df["LiedFirst"].astype(bool)

    # 1. One row per personality (+ slice)
    g_cols = ["Personality", "I/E", "N/S", "T/F", "J/P"]
    if slice_by:
        g_cols.insert(0, slice_by)

    per_type = df.groupby(g_cols, as_index=False)["LiedFirst"].mean()

    # 2. Two‑proportion tests for each dichotomy
    axes  = {"I/E": ("I", "E"),
             "N/S": ("N", "S"),
             "T/F": ("T", "F"),
             "J/P": ("J", "P")}
    lines = []
    for dim, (pos, neg) in axes.items():
        pos_vals = per_type.loc[per_type[dim] == pos, "LiedFirst"].values
        neg_vals = per_type.loc[per_type[dim] == neg, "LiedFirst"].values

        n_pos, n_neg = len(pos_vals), len(neg_vals)
        if min(n_pos, n_neg) == 0:
            lines.append(f"{dim}: not enough data (n_{pos}={n_pos}, n_{neg}={n_neg})")
            continue

        p = _prop_test(pos_vals.sum(), n_pos, neg_vals.sum(), n_neg)
        higher = pos if pos_vals.mean() > neg_vals.mean() else neg

        lines.append(
            f"{dim}: n_{pos}={n_pos}, n_{neg}={n_neg}  "
            f"{pos}={pos_vals.mean():.2%}, {neg}={neg_vals.mean():.2%}  "
            f"(p={p:.3g}) — {higher} more likely to lie first"
        )

    # 3. Store
    os.makedirs(plots_path, exist_ok=True)
    path = os.path.join(plots_path, f"liedfirst_pvalues_{label}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    return lines

                    
def plot_boxplot_score_all(df_agents, plots_path, df_type):
    df = df_agents.copy()
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    ax = sns.boxplot(
        data=df,
        x="Personality",
        y="TotalScore",
        order=personality_order,
        hue="Personality",
        palette=PALETTE_BY_TYPE
    )
    
    plt.axhline(y=21, color='g', linestyle='--', label='Max if both cooperate: 21')
    plt.axhline(y=45, color='r', linestyle='--', label='Max if one defects: 45')
    plt.axhline(y=7, color='b', linestyle='--', label='Max if both defect: 7')
    
    plt.legend(loc='upper center')
    ax.set_xlabel("")
    ax.set_ylabel("Total Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "score_boxplot_all.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot_score_per_dichotomy(df, plots_path, df_type):
    df = df.copy()
    sns.set(style="whitegrid", context="talk")

    # Define MBTI dichotomies and their levels
    dichotomy_pairs = [("I/E", ["I", "E"]),
                       ("N/S", ["N", "S"]),
                       ("T/F", ["T", "F"]),
                       ("J/P", ["J", "P"])]

    # Ensure output path exists
    os.makedirs(plots_path, exist_ok=True)

    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
    axes_flat = axes.flatten()

    for i, (dich, levels) in enumerate(dichotomy_pairs):
        ax = axes_flat[i]
        sns.boxplot(
            data=df,
            x=dich,
            y="TotalScore",
            order=levels,
            hue=dich,
            hue_order=levels,
            palette=[HUE_COLOUR[level] for level in levels],
            ax=ax
        )
        # Remove individual legends
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_xlabel(dich)
        ax.set_ylabel("Total Score" if i == 0 else "")

    plt.tight_layout()
    file_path = os.path.join(plots_path, "score_boxplot_per_dichotomy.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_violin_score_all(df, plots_path, df_type):
    df = df.copy()
    df['TotalScore'] = df['CumulativeScore'].apply(
        lambda x: x[-1] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan
    )

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=df,
        x="Personality",
        y="TotalScore",
        order=personality_order,
        hue="Personality",
        palette=PALETTE_BY_TYPE,
        inner='quartile'
    )
    plt.axhline(y=21, color='g', linestyle='--', label='Max both cooperate: 21')
    plt.axhline(y=45, color='r', linestyle='--', label='Max one defects: 45')
    plt.axhline(y=7,  color='b', linestyle='--', label='Max both defect: 7')
    plt.legend(loc='upper center')
    ax.set_xlabel("")
    ax.set_ylabel("Total Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "score_violin_all.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_violin_score_per_dichotomy(df, plots_path, df_type):
    df = df.copy()
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(10, 14), sharey=True)
    axes_flat = axes.flatten()
    dichotomies = ["I/E", "N/S", "T/F", "J/P"]

    for i, dich in enumerate(dichotomies):
        ax = axes_flat[i]
        levels = dich.split("/")
        sns.violinplot(
            data=df,
            x=dich,
            y="TotalScore",
            order=levels,
            hue=dich,
            hue_order=levels,
            palette=[HUE_COLOUR[level] for level in levels],
            inner='quartile',
            ax=ax
        )
        ax.set_xlabel(dich)
        ax.set_ylabel("Total Score" if i == 0 else "")
        if ax.get_legend():
            ax.get_legend().remove()

    # Plot for special personalities
    other = df[df["Personality"].isin(["NONE", "ALTRUISTIC", "SELFISH"])]
    ax5 = axes_flat[4]
    sns.violinplot(
        data=other,
        x="Personality",
        y="TotalScore",
        order=["NONE", "ALTRUISTIC", "SELFISH"],
        hue="Personality",
        palette={k: PALETTE_BY_TYPE[k] for k in ["NONE", "ALTRUISTIC", "SELFISH"]},
        inner='quartile',
        ax=ax5
    )
    ax5.set_xlabel("Other")
    ax5.set_ylabel("")
    if ax5.get_legend():
        ax5.get_legend().remove()

    # Hide empty subplot
    axes_flat[5].axis("off")

    plt.tight_layout()
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "score_violin_per_dichotomy.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_line_cumulative_score_all(df, plots_path, df_type):
    df = df.copy()
    # Create a dict with {Personality: [round0:[CumulativeScore], round1:[CumulativeScore], ...]}
    personality_cumulative_all = {}
    personality_cumulative_means = {}
    # Take the mean of the list of CumulativeScores for each round
    for personality in personality_order:
        personality_cumulative_all[personality] = []
    for index, row in df.iterrows():
        personality = row['Personality']
        if personality not in personality_order:
            continue
        cumulative_scores = row['CumulativeScore']
        for round_index, score in enumerate(cumulative_scores):
            if len(personality_cumulative_all[personality]) <= round_index:
                personality_cumulative_all[personality].append([])
            personality_cumulative_all[personality][round_index].append(score)
    for personality, scores in personality_cumulative_all.items():
        personality_cumulative_means[personality] = [np.mean(round_scores) for round_scores in scores]
    # Create a df with the means
    df_means = pd.DataFrame.from_dict(personality_cumulative_means, orient='index').T
    df_means.columns = personality_order
    df_means.index = [f"Round {i}" for i in range(len(df_means))]
    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df_means, palette="tab20", linewidth=1.5, dashes=False)

    # Legend outside the plot to the right
    plt.legend(
        title="Personality",
        bbox_to_anchor=(1.05, 1),  # x=1.05 pushes it outside
        loc='upper left',
        borderaxespad=0.
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Score")
    plt.xticks(rotation=45, ha="right")
    plt.title("Cumulative Score by Personality")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room on right for legend

    file_path = os.path.join(plots_path, "cumulative_line_score_all.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_truth_bar_all(df, plots_path, df_type):
    """
    Bar plot of the *mean* Truthfulness for every personality.
    """
    df = df.copy()

    means = (
        df.groupby("Personality", as_index=False)["Truthfulness"]
          .mean()
          .set_index("Personality")
          .reindex(personality_order)
    )

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=means.reset_index(),
        x="Personality",
        y="Truthfulness",
        order=personality_order,
        hue="Personality",
        palette=PALETTE_BY_TYPE
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean Truthfulness")
    plt.xticks(rotation=45, ha="right")
    plt.title("Mean Truthfulness by Personality")
    plt.tight_layout()

    os.makedirs(plots_path, exist_ok=True)
    fname = os.path.join(plots_path, "truth_bar_all.pdf")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_truth_box_all(df, plots_path, df_type):
    """
    Box‑and‑whisker plot of Truthfulness for every personality.
    """
    df = df.copy()
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df,
        x="Personality",
        y="Truthfulness",
        order=personality_order,
        hue="Personality",
        palette=PALETTE_BY_TYPE
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_xlabel("")
    ax.set_ylabel("Truthfulness")
    plt.xticks(rotation=45, ha="right")
    plt.title("Truthfulness Distribution by Personality")
    plt.tight_layout()

    os.makedirs(plots_path, exist_ok=True)
    fname = os.path.join(plots_path, "truth_box_all.pdf")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_truth_violin_dichotomy(df, plots_path, df_type):
    """
    Creates a 3×2 grid of violin plots (four dichotomies + 'NONE/ALTRUISTIC/SELFISH')
    for the Truthfulness variable.
    """
    df = df.copy()
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharey=True)
    axes_flat = axes.flatten()
    dichotomies = ["I/E", "N/S", "T/F", "J/P"]

    # 1–4 ─ each MBTI dichotomy
    for i, dich in enumerate(dichotomies):
        ax = axes_flat[i]
        levels = [lvl for lvl in dich.split("/") if lvl in df[dich].unique()]
        sns.violinplot(
            data=df,
            x=dich,
            y="Truthfulness",
            order=levels,
            hue=dich,
            hue_order=levels,
            palette=[HUE_COLOUR[lvl] for lvl in levels],
            ax=ax,
            inner="quartile"
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.set_xlabel(dich)
        ax.set_ylabel("Truthfulness" if i == 0 else "")

    # 5 ─ special group
    other = df[df["Personality"].isin(["NONE", "ALTRUISTIC", "SELFISH"])]
    ax5 = axes_flat[4]
    sns.violinplot(
        data=other,
        x="Personality",
        y="Truthfulness",
        order=["NONE", "ALTRUISTIC", "SELFISH"],
        hue="Personality",
        palette={k: PALETTE_BY_TYPE[k] for k in ["NONE", "ALTRUISTIC", "SELFISH"]},
        ax=ax5,
        inner="quartile"
    )
    leg = ax5.get_legend()
    if leg is not None:
        leg.remove()
    ax5.set_xlabel("Other")
    ax5.set_ylabel("")

    # Turn off unused subplot (index 5)
    axes_flat[5].axis("off")

    plt.tight_layout()
    os.makedirs(plots_path, exist_ok=True)
    fname = os.path.join(plots_path, "truth_violin_dichotomy.pdf")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_truth_violin_dichotomy_IE(df, plots_path, df_type):
    """
    Single violin plot comparing Truthfulness for the I/E dichotomy only.
    Rows where `I/E` is None/NaN are ignored.
    """
    df = df.copy()
    df = df[df["I/E"].isin(["I", "E"])].dropna(subset=["I/E"])
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(
        data=df,
        x="I/E",
        y="Truthfulness",
        order=["I", "E"],
        hue="I/E",
        palette=[HUE_COLOUR["I"], HUE_COLOUR["E"]],
        inner="quartile"
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_xlabel("I / E")
    ax.set_ylabel("Truthfulness")
    plt.title("Truthfulness by I/E Dichotomy")
    plt.tight_layout()

    os.makedirs(plots_path, exist_ok=True)
    fname = os.path.join(plots_path, "truth_violin_IE.pdf")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_defection_rate_bar_dichotomy(df_agents: pd.DataFrame, plots_path, df_type) -> str:
    """
    Bar plots of mean defection rate for each MBTI dichotomy (I/E, N/S, T/F, J/P).
    Saves <plots_path>/defection_rate_by_dichotomy.pdf and returns the filepath.
    """
    def list_to_rate(actions):
        if isinstance(actions, str):
            actions = ast.literal_eval(actions)
        return np.mean([str(a).strip().lower() == "defect" for a in actions])

    df = df_agents.copy()
    df["defection_rate"] = df["Actions"].apply(list_to_rate)

    group_cols = ["Personality", "I/E", "N/S", "T/F", "J/P"]
    mbti_means = (
        df.groupby(group_cols, as_index=False)["defection_rate"]
          .mean()
    )

    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes = axes.flatten()

    dichotomies = [("I/E", ("I", "E")),
                   ("N/S", ("N", "S")),
                   ("T/F", ("T", "F")),
                   ("J/P", ("J", "P"))]

    for ax, (col, order) in zip(axes, dichotomies):
        sns.barplot(
            data=mbti_means,
            x=col, y="defection_rate",
            order=list(order),
            hue=col,
            hue_order=order,
            palette=[HUE_COLOUR[o] for o in order],
            legend=False,
            errorbar="sd",
            ax=ax,
        )
        ax.set_title(f"Defection rate by {col}")
        ax.set_xlabel("")
        ax.set_ylabel("Mean defect rate")

    fig.tight_layout()
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "defection_rate_by_dichotomy.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return file_path


def plot_defection_rate_bar_all(df_agents: pd.DataFrame, plots_path: str | os.PathLike, df_type) -> str:
    def list_to_rate(actions):
        if isinstance(actions, str):
            actions = ast.literal_eval(actions)
        return np.mean([str(a).strip().lower() == "defect" for a in actions])

    df = df_agents.copy()
    df["defection_rate"] = df["Actions"].apply(list_to_rate)

    mbti_means = (
        df.groupby("Personality", as_index=False)["defection_rate"]
          .mean()
          .sort_values("defection_rate", ascending=False)
    )

    fig, ax = plt.subplots()
    sns.barplot(
        data=mbti_means,
        x="Personality", y="defection_rate",
        hue="Personality",
        order=personality_order,
        palette=PALETTE_BY_TYPE,
        legend=False,
        errorbar="sd",
        ax=ax,
    )

    ax.set_title("Mean defection rate by MBTI personality")
    ax.set_xlabel("")
    ax.set_ylabel("Mean defect rate")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    fig.tight_layout()
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "defection_rate_all.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return file_path

def plot_truth_strip_dichotomy(df_agents, plots_path, df_type):
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))

    # Get 16 colors and map them to personality types
    personalities = sorted(df_agents['Personality'].dropna().unique())
    palette_colors = sns.color_palette("tab20", n_colors=len(personalities))
    palette = dict(zip(personalities, palette_colors))  # Map personality -> color

    # Re-group
    df_grouped_game = df_agents.groupby(
        ['GameName', 'Personality', 'I/E', 'N/S', 'T/F', 'J/P']
    )['Truthfulness'].mean().reset_index()

    dichotomies = ['I/E', 'N/S', 'T/F', 'J/P']
    axes = axs.flatten()

    # Plot without subplot legends
    for ax, col in zip(axes, dichotomies):
        sns.stripplot(
            x=col, y='Truthfulness', hue='Personality', data=df_grouped_game,
            ax=ax, palette=palette, jitter=True, dodge=False, size=10, legend=False
        )
        ax.set_title(f'Truthfulness by {col}')

    # Manually create legend handles
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=palette[p], label=p, markersize=10)
        for p in personalities
    ]

    fig.legend(
        handles=handles,
        title='Personality',
        loc='center left',
        bbox_to_anchor=(0.85, 0.5),
        frameon=True
    )

    # Leave space on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Shrinks plotting area to 85% width

    plt.savefig(os.path.join(plots_path, 'truth_strip_dichotomy.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_truth_by_round(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    df_agents['Round'] = df_agents['CumulativeScore'].apply(lambda x: list(range(len(x))))
    df_agents['TruthfulByRound'] = df_agents.apply(lambda row: list(zip(row['Round'], row['Truthful'])), axis=1)

    df_exploded = df_agents.explode('TruthfulByRound')
    df_exploded['Round'] = df_exploded['TruthfulByRound'].apply(lambda x: x[0] if isinstance(x, tuple) else None)
    df_exploded['Truthful'] = df_exploded['TruthfulByRound'].apply(lambda x: x[1] if isinstance(x, tuple) else None)
    df_exploded = df_exploded.dropna(subset=['Round', 'Truthful'])
    df_exploded['Round'] = df_exploded['Round'].astype(int)

    sns.set(style="whitegrid", context="talk")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    dichotomies = [("I/E", ["I", "E"]),
                   ("N/S", ["N", "S"]),
                   ("T/F", ["T", "F"]),
                   ("J/P", ["J", "P"])]

    for ax, (dich, levels) in zip(axs.flat, dichotomies):
        sns.lineplot(
            data=df_exploded,
            x='Round', y='Truthful',
            hue=dich,
            hue_order=levels,
            palette=[HUE_COLOUR[level] for level in levels],
            ax=ax
        )
        ax.set_title(f'Average Truthfulness by Round - {dich}')
        ax.set_ylabel("Average Truthfulness")
        ax.set_xlabel("Round")

    plt.tight_layout()
    os.makedirs(plots_path, exist_ok=True)
    plt.savefig(os.path.join(plots_path, "truthfulness_by_round.pdf"), dpi=300)
    plt.close(fig)


def _legend_above(fig, title="", ncol=8):
    """
    Helper to build one legend, now above the whole figure.
    """
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        if ax.legend_:
            ax.legend_.remove()

    uniq = {lab: h for lab, h in zip(labels, handles)}
    full_labels = [LABEL_FULL[lab] for lab in uniq.keys()]

    fig.legend(
        uniq.values(), full_labels,
        title=title,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=ncol,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    
def _legend_mbti(fig):
    """
    Colour legend for MBTI dichotomies, centred above the axes.
    """
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
        if ax.legend_: ax.legend_.remove()

    uniq = {lab: h for lab, h in zip(labels, handles)}
    full = [LABEL_FULL[l] for l in uniq]
    fig.legend(
        uniq.values(), full, title="MBTI dichotomies",
        loc="lower center", bbox_to_anchor=(0.5, 1.19),
        frameon=False, ncol=8, handletextpad=.6, columnspacing=1.4,
    )

def _legend_alias(fig, alias_map, max_columns=4):
    from matplotlib.patches import Patch
    import math
    """Text-only legend mapping alias → full model name."""
    items = [f"{abbr} – {full}" for full, abbr in alias_map.items()]
    # Break into rows so it stays narrow enough
    rows = math.ceil(len(items) / max_columns)
    chunks = [items[i::rows] for i in range(rows)]
    lines = ["   •   ".join(chunk) for chunk in chunks]

    # Single invisible patch so matplotlib reserves the spot
    patch = Patch(facecolor="white", edgecolor="white", label="\n".join(lines))
    fig.legend(
        handles=[patch], loc="lower center",
        bbox_to_anchor=(0.5, 1.12), frameon=False,
        handlelength=0, handletextpad=0, fontsize="small",
    )


def plot_truthfulness_by_df_type_dichotomies(df_agents, plots_path, df_type):
    """
    Makes a 2×2 grid of barplots:
      • x-axis uses aliases (see MODEL_ALIAS)
      • top legend 1: alias → full model name
      • top legend 2: colour code for MBTI dichotomies
    """
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"

    # Add alias column
    df = df_agents.copy()
    alias_col = f"{slice_by}_alias"
    df[alias_col] = df[slice_by].map(MODEL_ALIAS).fillna(df[slice_by])

    # Plotting
    sns.set(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(22, 13))

    pairs = [("I/E", ["I", "E"]),
             ("N/S", ["N", "S"]),
             ("T/F", ["T", "F"]),
             ("J/P", ["J", "P"])]

    for i, (pair, hues) in enumerate(pairs, 1):
        ax = plt.subplot(2, 2, i)
        sns.barplot(
            x=alias_col, y="Truthfulness", hue=pair,
            data=df, palette=[HUE_COLOUR[h] for h in hues],
            hue_order=hues, ax=ax,
        )
        ax.set(xlabel=f"{slice_by} (alias)", ylabel="Truthfulness")
        ax.tick_params(axis="x", rotation=45)

    # Legends
    if df_type == "model":
        _legend_alias(fig, MODEL_ALIAS, max_columns=4)
    _legend_mbti(fig)

    # Leave space for legends (auto: plenty for two lines)
    fig.subplots_adjust(top=1.18)
    fig.tight_layout(rect=[0, 0, 1, 1.15])
    # Remove any super-title
    fig.suptitle("")
    fig.savefig(f"{plots_path}truthfulness_by_{slice_by}_dichotomies.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_truthfulness_by_df_type_IE(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"

    sns.set(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(15, 9))

    ax = plt.subplot(1, 1, 1)
    sns.barplot(
        x=slice_by, y="Truthfulness", hue="I/E",
        data=df_agents,
        palette=[HUE_COLOUR["I"], HUE_COLOUR["E"]],
        hue_order=["I", "E"],
        ax=ax,
    )
    ax.set(xlabel=f"{slice_by} name", ylabel="Truthfulness")
    ax.tick_params(axis="x", rotation=45)

    fig.subplots_adjust(top=0.99)
    _legend_above(fig, title="Introversion / Extraversion", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{plots_path}truthfulness_bar_I_E.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    

def plot_truthfulness_violin_by_df_typ(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"
    sns.set(style="whitegrid", context="talk")
    slices = df_agents[slice_by].unique()
    for slice_value in slices:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(
            x="I/E",
            y="Truthfulness",
            data=df_agents[df_agents[slice_by] == slice_value],
            palette=[HUE_COLOUR["I"], HUE_COLOUR["E"]],
            hue="I/E",
            order=["I", "E"],
            ax=ax,
        )
        # Remove any leftover legend
        if ax.legend_:
            ax.legend_.remove()
        ax.set(
            xlabel="I/E",
            ylabel="Truthfulness",
        )
        ax.tick_params(axis="x")
        plt.tight_layout()
        plt.savefig(f"{plots_path}truthfulness_violin_{slice_value}_I_E.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_mean_truthfulness_by_df_type(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    if df_type == "pd":
        return
    slice_by = "Model" if df_type == "model" else "GameName"
    sns.barplot(data=df_agents, x=slice_by, y='Truthfulness', palette='Set2',hue=slice_by)
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.xlabel(f'{slice_by}')
    plt.ylabel('Truthfulness Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'truthfulness_mean_all_{df_type}.pdf'), bbox_inches='tight')
    plt.close()


def plot_strategy_switches_dichotomy(df_agents, plots_path, df_type=None):
    """
    Generate a 2x2 grid of bar plots showing the mean number of strategy switches,
    grouped by MBTI dichotomies (I/E, N/S, T/F, J/P), using consistent MBTI color coding.
    """
    df = df_agents.copy()
    dichotomy_pairs = [("I/E", ["I", "E"]),
                       ("N/S", ["N", "S"]),
                       ("T/F", ["T", "F"]),
                       ("J/P", ["J", "P"])]

    os.makedirs(plots_path, exist_ok=True)

    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (dim, hues) in enumerate(dichotomy_pairs):
        if dim not in df.columns:
            continue

        summary = (
            df.groupby(dim, as_index=False)["Switches"]
              .mean()
              .rename(columns={"Switches": "MeanSwitches"})
        )

        sns.barplot(
            data=summary,
            x=dim,
            y="MeanSwitches",
            hue=dim,
            hue_order=hues,
            palette=[HUE_COLOUR[h] for h in hues],
            ax=axes[i],
        )

        axes[i].set_ylabel("Mean Strategy Switches")
        axes[i].set_title(f"Strategy Switches by {dim}")
        if axes[i].get_legend():
            axes[i].legend_.remove()

    plt.tight_layout()
    file_path = os.path.join(plots_path, "strategy_switches_by_dichotomy_grid.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    
def plot_strategy_switches_by_df_type_TF(df_agents, plots_path, df_type):
    if df_type == "pd":
        return

    df_agents = df_agents.copy()
    slice_by = "Model" if df_type == "model" else "GameName"

    os.makedirs(plots_path, exist_ok=True)

    for slice_name in df_agents[slice_by].unique():
        slice_df = df_agents[df_agents[slice_by] == slice_name]
        t_switches = slice_df.loc[slice_df['T/F'] == 'T', 'Switches']
        f_switches = slice_df.loc[slice_df['T/F'] == 'F', 'Switches']

        t_counts = t_switches.value_counts().reindex(range(7), fill_value=0)
        f_counts = f_switches.value_counts().reindex(range(7), fill_value=0)

        plt.figure(figsize=(7, 5))
        sns.barplot(x=t_counts.index, y=t_counts.values,
                    color=HUE_COLOUR["T"], label='T')
        sns.barplot(x=f_counts.index, y=-f_counts.values,
                    color=HUE_COLOUR["F"], label='F')

        max_y = max(t_counts.values.max(), f_counts.values.max())
        plt.ylim(-max_y - 2, max_y + 2)
        plt.axhline(0, color='black', linewidth=0.8)

        plt.xlabel('Number of Strategy Switches')
        plt.ylabel('Frequency')
        plt.legend(title='T/F', loc='upper right')
        plt.tight_layout()

        filename = f'strategy_switches_{slice_name}_TF.pdf'
        plt.savefig(os.path.join(plots_path, filename), dpi=300)
        plt.close()

        
def plot_who_lied_first_count_all(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()
    max_possible_lies = df_agents["LiedFirst"].sum()
    # Compute fraction of episodes where each agent lied first
    summary = (
        df_agents
        .groupby("Personality", as_index=False)["LiedFirst"]
        .mean()
    )

    # Plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(
        data=summary,
        x="Personality", y="LiedFirst",
        hue="Personality",
        palette=PALETTE_BY_TYPE,
        ax=ax,
        order=personality_order,
    )
    ax.set_ylabel("Proportion Lied First")
    ax.tick_params(axis='x', rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')
    # Set y limit
    ax.set_ylim(0, 1)

    # Store
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "who_lied_first.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_who_lied_first_proportion_dichotomy(df_agents, plots_path, df_type=None):
    """
    Generate a 2x2 grid of bar plots showing the proportion of agents who lied first,
    grouped by MBTI dichotomies (I/E, N/S, T/F, J/P), using consistent MBTI color coding.
    """
    df = df_agents.copy()
    dichotomy_pairs = [("I/E", ["I", "E"]),
                       ("N/S", ["N", "S"]),
                       ("T/F", ["T", "F"]),
                       ("J/P", ["J", "P"])]

    os.makedirs(plots_path, exist_ok=True)

    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (dim, hues) in enumerate(dichotomy_pairs):
        if dim not in df.columns:
            continue

        summary = (
            df.groupby(dim, as_index=False)["LiedFirst"]
              .mean()
              .rename(columns={"LiedFirst": "ProportionLiedFirst"})
        )

        sns.barplot(
            data=summary,
            x=dim,
            y="ProportionLiedFirst",
            hue=dim,
            hue_order=hues,
            palette=[HUE_COLOUR[h] for h in hues],
            ax=axes[i],
        )

        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel("Proportion Lied First")
        axes[i].set_title(f"Lied First by {dim}")
        if axes[i].get_legend():
            axes[i].legend_.remove()

    plt.tight_layout()
    file_path = os.path.join(plots_path, "lied_first_by_dichotomy_grid.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_who_lied_first_IE_TF(df_agents, plots_path, df_type):
    df_agents = df_agents.copy()

    # I/E plot
    summary_ie = (
        df_agents
        .groupby("I/E", as_index=False)["LiedFirst"]
        .sum()
    )

    fig, ax = plt.subplots()
    sns.barplot(
        data=summary_ie,
        x="I/E", y="LiedFirst",
        hue="I/E",
        hue_order=["I", "E"],
        palette=[HUE_COLOUR["I"], HUE_COLOUR["E"]],
        ax=ax
    )
    ax.set_ylabel("Count Lied First")
    if ax.get_legend():
        ax.get_legend().remove()
    os.makedirs(plots_path, exist_ok=True)
    file_path = os.path.join(plots_path, "who_lied_first_IE.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # T/F plot
    summary_tf = (
        df_agents
        .groupby("T/F", as_index=False)["LiedFirst"]
        .sum()
    )

    fig, ax = plt.subplots()
    sns.barplot(
        data=summary_tf,
        x="T/F", y="LiedFirst",
        hue="T/F",
        hue_order=["T", "F"],
        palette=[HUE_COLOUR["T"], HUE_COLOUR["F"]],
        ax=ax
    )
    ax.set_ylabel("Count Lied First")
    if ax.get_legend():
        ax.get_legend().remove()
    file_path = os.path.join(plots_path, "who_lied_first_TF.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_from_list(df, plots_path, plot_functions, df_type=None):
    for func in plot_functions:
        try:
            func(df, plots_path, df_type)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            continue
        
def stripplot_test():
    df_type = 'model'
    df, plots_path = import_df_and_plot_path(df_type)
    df_agents_og = import_df_agents(df)
    df_agents = df_agents_og.copy()
    df_agent_functions = [plot_truth_strip_dichotomy]
    # Run the plotting functions
    plot_from_list(df_agents, plots_path, df_agent_functions, df_type)
        
# Run whats runnable
def PD_plots():
    df_type = 'pd'
     # Set the path to save the plots
    df, plots_path = import_df_and_plot_path(df_type)

    df_agents_og = import_df_agents(df)
    # Possible plotting functions
    # df_functions = [
    #     get_liedfirst_pvalues,
    # ]
    df_functions=[]
    df_agent_functions = [
        plot_truth_bar_all,
        plot_truth_box_all,
        plot_truth_violin_dichotomy,
        plot_truth_violin_dichotomy_IE,
        plot_boxplot_score_all,
        plot_boxplot_score_per_dichotomy,
        plot_violin_score_all,
        plot_violin_score_per_dichotomy,
        plot_line_cumulative_score_all,
        plot_defection_rate_bar_all,
        plot_defection_rate_bar_dichotomy,
        get_defection_rates_dichos,
        plot_truth_strip_dichotomy,
        get_truthfulness_dichos,
        plot_truth_by_round,
        get_liedfirst_pvalues,
        plot_who_lied_first_count_all,
        plot_who_lied_first_proportion_dichotomy,
        plot_who_lied_first_IE_TF,
        get_strategy_switch_pvalues,
    ]
    # Run the plotting functions
    plot_from_list(df, plots_path, df_functions, df_type)
    plot_from_list(df_agents_og, plots_path, df_agent_functions, df_type)


def model_plots():
    df_type = 'model'
    # Set the path to save the plots
    df, plots_path = import_df_and_plot_path(df_type)

    df_agents_og = import_df_agents(df)
    df_agent_functions = [
        plot_truth_bar_all,
        plot_truth_box_all,
        plot_truth_violin_dichotomy,
        plot_truth_violin_dichotomy_IE,
        plot_boxplot_score_all,
        plot_boxplot_score_per_dichotomy,
        plot_violin_score_all,
        plot_violin_score_per_dichotomy,
        plot_line_cumulative_score_all,
        plot_defection_rate_bar_all,
        plot_defection_rate_bar_dichotomy,
        get_defection_rates_dichos,
        plot_truth_strip_dichotomy,
        get_truthfulness_dichos,
        plot_truth_by_round,
        plot_truthfulness_by_df_type_dichotomies,
        plot_truthfulness_by_df_type_IE,
        plot_truthfulness_violin_by_df_typ,
        plot_mean_truthfulness_by_df_type,
        plot_strategy_switches_dichotomy,
        plot_strategy_switches_by_df_type_TF,
        get_strategy_switch_pvalues,
        get_liedfirst_pvalues,
        plot_who_lied_first_count_all,
        plot_who_lied_first_proportion_dichotomy,
        plot_who_lied_first_IE_TF,
    ]
    # Run the plotting functions
    plot_from_list(df_agents_og, plots_path, df_agent_functions, df_type)

def game_plots():
    df_type = 'game'
    # Set the path to save the plots
    df, plots_path = import_df_and_plot_path(df_type)

    df_agents_og = import_df_agents(df)
    
    df_agent_functions = [
        plot_truth_bar_all,
        plot_truth_box_all,
        plot_truth_violin_dichotomy,
        plot_truth_violin_dichotomy_IE,
        plot_truth_strip_dichotomy,
        get_truthfulness_dichos,
        plot_truth_by_round,
        plot_truthfulness_by_df_type_dichotomies,
        plot_truthfulness_by_df_type_IE,
        plot_truthfulness_violin_by_df_typ,
        plot_mean_truthfulness_by_df_type,
        plot_strategy_switches_dichotomy,
        plot_strategy_switches_by_df_type_TF,
        get_strategy_switch_pvalues,
        get_liedfirst_pvalues,
        plot_who_lied_first_count_all,
        plot_who_lied_first_proportion_dichotomy,
        plot_who_lied_first_IE_TF,
    ]
    plot_from_list(df_agents_og, plots_path, df_agent_functions, df_type)

if __name__ == '__main__':
    # Set the style of seaborn
    sns.set(style="whitegrid")
    sns.set_palette("Set2")

    # Set the font size for all plots
    plt.rcParams.update({'font.size': 14})

    # Set the figure size for all plots
    plt.rcParams['figure.figsize'] = [10, 6]

    PD_plots()
    model_plots()
    game_plots()
