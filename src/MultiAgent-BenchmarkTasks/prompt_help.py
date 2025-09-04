# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import json

from hf_datasets import get_dataset

def get_few_shots_task_prompt(dataset_name: str, config_name: str):
    """
    Takes in a dataset to create a prompt to add examples to the task
    """
    task_prompt=""
    json_file = 'data/task_descriptions.json'
    dataset = get_dataset(dataset_name, config_name)
    training_data = dataset['train'] # Get this data to get examples (few shots learning)
    # Get 3 examples with labels
    few_shots_examples :str
    for i in range(3):
        example = training_data[i]
        few_shots_examples += f"\n## TASK {i+1}: {example['text']}"
        few_shots_examples += f"\n### SOLUTION: {example['label']}"
        
    with open(json_file) as f:
        task_descriptions = json.load(f)
        task_prompt+=f"""# Task Description:\n{task_descriptions[dataset_name][config_name]}\n# Examples:{few_shots_examples}
    """
    return task_prompt

def get_system_prompt_supervisor(agent_names):
    return f"""You are a supervisor tasked with managing a conversation between the following workers: {agent_names}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and you will be the judge, you do not have to ask all the agents, once you think you had enough advice you have to respond with FINISH, your reasoning: SOLUTIONEXPLANATION and your final SOLUTION: followed by one of the solution optinos given in the task description. """

def get_agent_reminder(agent_name):
    return f"# Important notes: 1. Your name is {agent_name}.\n 2. You HAVE TO make sure that your answers and discussion align with your personality: State your answer based on your prompted personality in every message.\n." 

def get_agent_scratchpad_reminder():
    return "You made up your mind in your personal scratchpad, use this information as well."

def get_mbti_prompt(mbti_type: str):
    try:
        with open('../../priming/priming_without_mention_of_mbti_different_none.json') as f:
            data = json.load(f)
            return data[mbti_type]
    except KeyError:
        print("Only give MBTI personalities or NONE or EXPERT")

def get_network_agent_prompt(MBTI_types):
    """
    This function will return the prompt for the blackboard agents.
    Args:
        MBTI_types (list): The list of MBTI types.
    """
    prompt = f"""You are one of the following agents: {MBTI_types}.
You will be tasked with solving a problem with eachother, so do not be shy to hear everyone out, or have an actual discussion with one another. Once you deem the conversation to be done, end the discussion by giving the solution.  
You will be given the task description and the messages from each agent. 
You have to:
1. state your opinion on this matter 
2. route the conversation
    - to another agent.
    - or end the discussion and write out the solution
"""
    return prompt

def get_scratchpad_agent_prompt(MBTI_types):
    """
    This function will return the prompt for the blackboard agents.
    Args:
        MBTI_types (list): The list of MBTI types.
    """
    prompt = f"""You are one of the following agents: {MBTI_types}.
Before interacting with the other agents, you must first fill in your personal scratchpad with your thoughts and reasoning about the task. Use this scratchpad to organize your ideas and ensure your contributions are well thought out.
Once your scratchpad is complete, you will be tasked with solving a problem with each other, so do not be shy to hear everyone out or have an actual discussion with one another. Once you deem the conversation to be done, end the discussion by giving the solution.  
You will be given the task description and the messages from each agent. 
You have to:
1. state your opinion on this matter based on your scratchpad
2. route the conversation
    - to another agent.
    - or end the discussion and write out the solution
"""
    return prompt
