# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import json
import logging
import os

from backers.models import get_model
from flows import invoke_graph_one_shot
from hf_datasets import get_dataset
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from prompt_help import get_mbti_prompt
from typing import Literal
from typing_extensions import TypedDict

def create_reduce_node_function():
    """
    Adds the majority vote result according to the state
    """
    def supervisor_node_function(state) -> Command[Literal["__end__"]]:
        messages = state["messages"]
        votes = []
        for message in messages:
            if hasattr(message, 'message_type') and message.message_type == 'vote':
                votes.append(message.content)
        # Count the votes
        vote_counts = {}
        for vote in votes:
            if vote not in vote_counts:
                vote_counts[vote] = 0
            vote_counts[vote] += 1
        majority_vote = max(vote_counts, key=vote_counts.get)
        # Add the majority vote to the state
        state["messages"].append(AIMessage(content=majority_vote, name="supervisor", message_type="solution"))
        return Command(goto=END)
    return supervisor_node_function


def create_mbti_voter_node_function(agent_name, agent_type, structure, model):
    agents_personality_prompt = SystemMessage(get_mbti_prompt(agent_type), name="system")
    def mbti_node_voter_function(state) -> Command[Literal["ballot_agent"]]:
        temporary_messages = state["messages"].copy()
        temporary_messages.insert(0, agents_personality_prompt)
        response = model.with_structured_output(structure).invoke(temporary_messages)
        vote = response["SOLUTION"]
        logging.info(f"mbi_type: {agent_type} voted: {vote}")
        explanation = response["SOLUTION EXPLANATION"]
        logging.info(f"mbi_type: {agent_type} explanation: {explanation}")
        # Adding the info as "agent type" because it doesn't affect the ballot agent anyway
        return Command(
            update={
                "messages": [
                    AIMessage(content=explanation, name=agent_type, message_type="explanation"),
                    AIMessage(content=vote, name=agent_type, message_type="vote")
                ]
            },
            goto="ballot_agent",
        )
    return mbti_node_voter_function


def solve_task(dataset_name, dataset_config, model_name: str, agent_types_strings: list, n_tasks = -1, runs = 1):
    """
    Each agent gets one vote and the task will be solved by the majority vote.

    Args:
        model_name (str): Name of the language model.
        agent_types_strings (List[str]): List of the MBTI types for the agents.
    """
    model = get_model(model_name = model_name, max_new_tokens = 100)
    structure = TypedDict('State', {'SOLUTION': str, 'SOLUTION EXPLANATION': str}, total=True)
    agents_info = {f"agent{i}": agent_type for i, agent_type in enumerate(agent_types_strings)}
    agent_nodes = []
    for agent_name, agent_type in agents_info.items():
        agent_nodes.append((agent_name, create_mbti_voter_node_function(agent_name, agent_type, structure, model))
    )
    reducer_function = create_reduce_node_function()
    # Build graph
    builder = StateGraph(MessagesState)
    for agent_name, agent_callable in agent_nodes:
        builder.add_node(agent_name, agent_callable)
        builder.add_edge(START, agent_name)
    builder.add_node("ballot_agent", reducer_function)
    builder.add_edge("ballot_agent", END)
    
    # Compile graph
    graph = builder.compile()
    logging.info(graph.get_graph().draw_mermaid())
    # Invoke the graph
    invoke_graph_one_shot(graph, dataset_name, dataset_config, agent_types_strings, "vote", n_tasks, runs)    
