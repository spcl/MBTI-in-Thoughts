# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import json
import os
import pandas as pd

from hf_datasets import get_dataset
from langchain-community.callbacks import get_openai_callback
from langchain-core.messages import SystemMessage, AIMessage, HumanMessage

def get_cb_dict(cb):
    # Callback for cost
    cb_dict = {}
    for line in str(cb).split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            cb_dict[key.strip()] = value.strip()
    return cb_dict

def invoke_graph_one_shot(graph, dataset_name, dataset_config, agent_types_strings, solving_method, n_tasks = -1, runs = 1):
    tasks, labels, options  = get_dataset(dataset_name = dataset_name, dataset_config = dataset_config, n_tasks = n_tasks) # Because we are using the testing dataset for eval
    for j in range(runs):
        df = pd.DataFrame(columns=["prediction", "label", "task", "prompt_tokens", "completion_tokens", "llm_requests", "cost (USD)", "options","state"])
        current_total_cost = 0
        total_size = len(tasks)
        for i, (task, label, option) in enumerate(zip(tasks, labels, options)):
            print(f"currently on run {j} and task {i}/{total_size}")
            config = {"configurable": {"thread_id": str(i)}}
            # Form task with prompt helper by replacing 'text' with task
            task_prompt = task
            print(f"NEW TASK: {task_prompt}, options: {option}")
            try:
                with get_openai_callback() as cb:
                    state = graph.invoke({"messages": [HumanMessage(task_prompt),HumanMessage("SOLUTION options:" + str(option))]},
                                         config, 
                                         subgraphs=True)
                cb_dict = get_cb_dict(cb)
                prompt_tokens = int(cb_dict["Prompt Tokens"])
                completion_tokens = int(cb_dict["Completion Tokens"])
                llm_requests = int(cb_dict["Successful Requests"])
                cost = float(cb_dict["Total Cost (USD)"].replace('$', ''))
                prediction = [state[1]["messages"][-1].content]
                label = label if isinstance(label,list) else [label]  # multitarget
                df.loc[i] = [prediction, label, task.replace("\n", " "), prompt_tokens, completion_tokens, llm_requests, cost, option, state]
                current_total_cost += cost
                print(f"Prediction: {prediction}, Label: {label}, Current Total Cost: {current_total_cost}")
            except Exception as e:
                import traceback
                print("Encountered an error, skipping question")
                print("Error details:", e)
                traceback.print_exc()
                df.loc[i] = ["error", label, task.replace("\n", " "), "NA", "NA", "NA", "NA", option, state]
            if current_total_cost > 3:
                print(f"Current total cost: {current_total_cost}, stopping")
                break    
        os.makedirs(f"data/results/{dataset_name}/{dataset_config}/{solving_method}", exist_ok=True)
        agent_types_str = "_".join(agent_types_strings)
        file_name = f"data/results/{dataset_name}/{dataset_config}/{solving_method}/{agent_types_str}_run_{j}.csv"
        open(file_name, 'a').close()
        df.to_csv(file_name, index=False)
