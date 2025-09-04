# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import argparse
import os

from hf_datasets import list_configs

def verify_args(args: argparse.Namespace) -> argparse.Namespace:
    # Make sure run is positive
    if args.runs < 1:
        raise ValueError("runs must be positive")
    return args

# Create argparse solver names from files that start with "solve_" and end with ".py"
def get_solver_names():
    return [f.replace("solve_", "").replace(".py", "") for f in os.listdir(".") if f.startswith("solve_") and f.endswith(".py")]

# Fetch dataset configurations dynamically
def get_configs(dataset_name):
    return list_configs(dataset_name)

def get_model_names():
    models = ["gpt-4o-mini"]
    return models

def main(args):
    # Call function from the solver file
    print(args)
    solver_module = __import__("solve_" + args.solver, fromlist=['solve_task' + args.solver])
    dataset_name = args.dataset
    dataset_config = args.dataset_config
    model_name = args.model
    agent_types_strings = args.agent_types
    n_tasks = args.n_tasks
    runs = args.runs
    
    args = verify_args(args)
    
    agent_types_strings = args.agent_types if isinstance(args.agent_types, list) else args.agent_types.split(',')
    agent_types_strings.sort()
    
    solver_module.solve_task(dataset_name=dataset_name,
                             dataset_config=dataset_config,
                             model_name=model_name,
                             agent_types_strings=agent_types_strings,
                             n_tasks=n_tasks,
                             runs=runs)

if __name__ == "__main__":
    # Step 1: Initial parser to get the dataset
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--solver', type=str, required=True, help='method to solve the task', choices=get_solver_names())
    parser.add_argument('--dataset', type=str, required=True, help='task dataset', choices=["Blablablab/SOCKET", "test", "hails/bigbench"])
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--n_tasks', type=int, default=-1, help='number of tasks to solve')
    parser.add_argument('--runs', type=int, default=1, help='number of runs')
    parser.add_argument('--agent_types', type=str, required=True, nargs='+', help='list of agent types: INTJ ENFP ISTP')

    # Initial parsing to get the dataset
    args, unknown = parser.parse_known_args()
    dataset_name = args.dataset

    # Step 2: Dynamically update the parser for dataset_config
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='a config of the Dataset',
                        choices=get_configs(dataset_name) + ["test"])

    # Final parsing with the updated parser
    args = parser.parse_args()
    main(args)
