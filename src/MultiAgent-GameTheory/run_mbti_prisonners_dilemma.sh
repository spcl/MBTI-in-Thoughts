#!/bin/bash

# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

personas=("NONE" "ISTJ" "ISFJ" "INFJ" "INTJ" "ISTP" "ISFP" "INFP" "INTP" "ESTP" "ESFP" "ENFP" "ENTP" "ESTJ" "ESFJ" "ENFJ" "ENTJ" "ALTRUISTIC" "SELFISH")
nr_of_rounds=7
model_id="gpt-4o-mini-2024-07-18"
game_name="prisoners_dilemma"

total_iterations=$(( (${#personas[@]} * (${#personas[@]} + 1)) / 2 ))
current_iteration=0

for i in "${!personas[@]}"; do
    for ((j=i; j<${#personas[@]}; j++)); do  # Start from i, allowing (A, A)
        agent_1="${personas[$i]}"
        agent_2="${personas[$j]}"
        current_iteration=$((current_iteration + 1))
        echo "Iteration $current_iteration of $total_iterations: $agent_1 vs $agent_2"
        python main.py --model_id $model_id --rounds $nr_of_rounds --agent_1_persona "$agent_1" --agent_2_persona "$agent_2" --game_name $game_name
    done
done
