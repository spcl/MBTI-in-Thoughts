#!/bin/bash

# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

solvers=("blackboard_independent" "vote" "blackboard_independent_scratchpad")
datasets=("hails/bigbench" "hails/bigbench" "hails/bigbench" "hails/bigbench" "Blablablab/SOCKET" "Blablablab/SOCKET")
dataset_configs=("causal_judgment_zero_shot" "dark_humor_detection_zero_shot"  "simple_ethical_questions_zero_shot" "disambiguation_qa_zero_shot" "complaints" "stanfordpoliteness")
agent_types_configs=(
    "INFJ ENTP ISFP"
    "ESFJ INTP ENFJ"
    "ISTP INFP ISFJ"
    "ESTP INTP ISFJ"
    "NONE NONE NONE"
    "EXPERT EXPERT EXPERT"
)

# Iterate over solvers
for solver in "${solvers[@]}"; do
    # Iterate over dataset_configs and their corresponding datasets
    for i in "${!dataset_configs[@]}"; do
        dataset="${datasets[$i]}"
        dataset_config="${dataset_configs[$i]}"
        # Iterate over agent types
        for agent_types in "${agent_types_configs[@]}"; do
            # Run the command
            python run_experiment.py \
                --solver "$solver" \
                --dataset "$dataset" \
                --dataset_config "$dataset_config" \
                --model gpt-4o-mini \
                --n_tasks -1 \
                --runs 1 \
                --agent_types $agent_types
        done
    done
done
