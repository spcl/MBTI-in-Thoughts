# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import pandas as pd

from datasets import load_dataset, get_dataset_config_names
from dotenv import load_dotenv

load_dotenv()

def keep_n_of_dataset(tasks, labels, options, n):
    return tasks[:n], labels[:n], options[:n]

def get_dataset(dataset_name, dataset_config: str, train_bool: bool = False, n_tasks = -1) -> tuple: # n-1 means the whole dataset
    """
    Get dataset from huggingface
    """
    if dataset_name == "hails/bigbench":
        dataset = load_dataset(dataset_name, dataset_config)["train" if train_bool else "default"]
        tasks = dataset['inputs']
        labels = dataset['targets']
        options = dataset['multiple_choice_targets']
        try:
            description = dataset['description']
            for (task, description) in zip(tasks, description):
                task = f"{description}: {task}"
        except:
            pass
        if n_tasks != -1:
            tasks, labels, options = keep_n_of_dataset(tasks, labels, options, n_tasks)
        # Print total size of dataset:
        print(f"Total size of dataset: {len(tasks)}")
        return tasks, labels, options
    
    task_descriptions_dict = pd.read_json("data/task_descriptions.json")
    #---------------------------------------------------------------------------------------------------------------------
    if dataset_name =="test":
        # Path to the dataset directory
        dataset_dir = "data/test_dataset/" # TODO: rename the testing dataset to avoid confusion of the concept

        # Specify files explicitly for train and test
        data_files = {
            "train": f"{dataset_dir}/train.json",
            "test": f"{dataset_dir}/test.json"
        }
        datn_tasksaset = load_dataset("json", data_files=data_files)["train" if train_bool else "test"]
    #---------------------------------------------------------------------------------------------------------------------
    elif dataset_name == "Blablablab/SOCKET": # TODO: maybe manipulate this with a dataframe to speed up the process
        dataset = load_dataset(dataset_name, dataset_config, )["train" if train_bool else "test"]
        
    tasks = dataset['text'] #TODO dynamically adapt according to task_description
    labels = dataset['label']
    options = []
    options_list = task_descriptions_dict[dataset_name][dataset_config]["options"] # This is a list of options for all the tasks
    for i, task in enumerate(tasks):
        tasks[i] = task_descriptions_dict[dataset_name][dataset_config]["question"].replace("text", task)
        options.append(options_list)
        labels[i] = options[i][labels[i]] # Convert label to string
    
    if n_tasks != -1:
        tasks, labels, options = keep_n_of_dataset(tasks, labels, options, n_tasks)
    print(f"Total size of dataset: {len(tasks)}")
    return tasks, labels, options

def list_configs(dataset_name: str):
    if dataset_name == "Blablablab/SOCKET":
        return get_dataset_config_names(dataset_name, trust_remote_code=True)
    return get_dataset_config_names(dataset_name)
