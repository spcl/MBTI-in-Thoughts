#!/bin/bash

# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <total_stories> <number_of_executions> <round_number> [dataset_type] [top_upvoted_csv] [eval_model_name] [binary_evaluation]"
    exit 1
fi

# Get the total stories, number of executions, and round number
total_stories=$1
n=$2
round_number=$3

# Optional parameters with default values
dataset_type=${4:-"top_upvoted"}
top_upvoted_csv=${5:-"top_stories_batch_1.csv"}
eval_model_name=${6:-"Qwen/Qwen2.5-14B-Instruct"}
binary_evaluation=${7:-"false"}

# Check if total_stories is a positive integer
if ! [[ "$total_stories" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Please provide a positive integer for the total stories."
    exit 1
fi

# Check if n is a positive integer
if ! [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Please provide a positive integer for the number of executions."
    exit 1
fi

# Check if round_number is a positive integer
if ! [[ "$round_number" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Please provide a positive integer for the round number."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Calculate range per execution
range_per_execution=$((total_stories / n))

# Execute the script n times with appropriate parameters, in parallel
for ((i=1; i<=n; i++)); do
    from_idx=$(( (i-1) * range_per_execution ))
    to_idx=$(( i * range_per_execution ))
    
    # For the last execution, ensure we cover up to the end of the range
    if [ $i -eq $n ]; then
        to_idx=$((total_stories))
    fi
    
    plot_save_path="plots_round_${round_number}_part_${i}"
    evaluation_save_path="evaluations_round_${round_number}_part_${i}"
    
    echo "Starting execution $i: Running with from_idx=$from_idx, to_idx=$to_idx, plot_save_path=$plot_save_path, evaluation_save_path=$evaluation_save_path, dataset_type=$dataset_type, top_upvoted_csv=$top_upvoted_csv"
    
    # Run the Python script in the background with full path
    python "$SCRIPT_DIR/evaluate_stories.py" \
        --load_path "round_${round_number}" \
        --plot_save_path "$plot_save_path" \
        --evaluation_save_path "$evaluation_save_path" \
        --from_idx $from_idx \
        --to_idx $to_idx \
        --dataset_type "$dataset_type" \
        --top_upvoted_csv "$top_upvoted_csv" \
        --eval_model_name "$eval_model_name" \
        --binary_evaluation "$binary_evaluation" &
    
    # Store the process ID
    pids[$i]=$!
    
    echo "Execution $i started with PID ${pids[$i]}."
done

# Wait for all background processes to finish
echo "Waiting for all processes to complete..."
for pid in ${pids[*]}; do
    wait $pid
    echo "Process with PID $pid completed."
done

echo "All executions completed."
