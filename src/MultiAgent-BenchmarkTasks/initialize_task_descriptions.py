# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import ast
import json
import pandas as pd

from hf_datasets import list_configs

def update_task_descriptions():
    """
    Update the task descriptions for the tasks in the dataset. Creates a dictionary with dataset name as key and task-names as subkeys with a question and options for the task.

    {
        "dataset_name": {
            "task_name": {
                "type": "type", #optional
                "question": "question",
                "options": ["option1", "option2"]
            }
        }
    } 
    """
    task_descriptions = {}
    # 1. Add test dataset
    task_descriptions_test = {}
    dataset_name = "test"
    task_name = "test"
    type = "CLS"
    question = "For the sentence: \"{text}\", which action should be taken?"
    options = ["A", "B", "C", "D"]
    task_descriptions_test[task_name] = {
        "type": type,
        "question": question,
        "options": options
    }
    task_descriptions[dataset_name] = task_descriptions_test
    #---------------------------------------------------------------------------------------------------------------------
    #2. Add SOCKET dataset
    socket_csv = pd.read_csv("https://raw.githubusercontent.com/minjechoi/SOCKET/refs/heads/main/experiments/zeroshot/socket_prompts.csv")
    socket_adapted = socket_csv[["task", "type", "question","options"]]
    dataset_name = "Blablablab/SOCKET"
    task_descriptions_socket = {}
    for _, row in socket_adapted.iterrows():
        task_name = row["task"]
        task_descriptions_socket[task_name] = {
            "type": row["type"],
            "question": row["question"],
            "options": [] if pd.isna(row["options"]) else ast.literal_eval(row["options"]) if row["options"] != "NaN" else []
        }
    task_descriptions[dataset_name] = task_descriptions_socket
    #---------------------------------------------------------------------------------------------------------------------
    #3. Add BIG-bench dataset
    dataset_name = "hails/bigbench"
    task_descriptions_bigbench = {}
    task_names = list_configs(dataset_name)
    for task_name in task_names:
        task_descriptions_bigbench[task_name] = {
            "question": "NaN",
            "options": "NaN"
        }
    task_descriptions[dataset_name] = task_descriptions_bigbench
    with open("data/task_descriptions.json", "w") as f:
        json.dump(task_descriptions, f, indent=4)
