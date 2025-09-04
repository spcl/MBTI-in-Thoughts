#!/bin/bash

# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

# Check if required arguments are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <total_range> <number_of_executions> <temperature> <output_name> <model_name>"
    exit 1
fi

# Get the total range, number of executions, and temperature
total_range=$1
n=$2
temperature=$3
output_name=$4
model_name=$5

# Check if total_range is a positive integer
if ! [[ "$total_range" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Please provide a positive integer for the total range."
    exit 1
fi

# Check if n is a positive integer
if ! [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Please provide a positive integer for the number of executions."
    exit 1
fi

# Check if temperature is a valid number
if ! [[ "$temperature" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Error: Please provide a valid number for temperature."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Calculate range per execution
range_per_execution=$((total_range / n))

# Execute the script n times with appropriate parameters, in parallel
for ((i=1; i<=n; i++)); do
    generate_from=$(( (i-1) * range_per_execution ))
    generate_to=$(( i * range_per_execution - 1 ))
    
    # For the last execution, ensure we cover up to the end of the range
    if [ $i -eq $n ]; then
        generate_to=$((total_range - 1))
    fi
    
    output_dir="${output_name}_part_$i"
    
    echo "Starting execution $i: Running with generate_from=$generate_from, generate_to=$generate_to, temperature=$temperature, output_dir=$output_dir, model_name=$model_name"
    
    # Run the Python script in the background with full path
    python "$SCRIPT_DIR/write_stories.py" --generate_from $generate_from --generate_to $generate_to --temperature $temperature --output_dir "$output_dir" --model_name "$model_name" &
    
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
