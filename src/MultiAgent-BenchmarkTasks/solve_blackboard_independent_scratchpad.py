# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import operator

from flows import invoke_graph_one_shot
from backers.models import get_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_core.messages.utils import AnyMessage
from langgraph.graph import MessagesState, START, END, StateGraph, add_messages
from langgraph.types import Command, Send
from prompt_help import get_mbti_prompt, get_scratchpad_agent_prompt, get_agent_reminder, get_agent_scratchpad_reminder
from pydantic import BaseModel
from random import choice
from typing import Literal, Dict, Optional, List, Annotated
from typing_extensions import TypedDict

class Scratchpad(BaseModel):
    """
    Make up your mind about the solution
    """
    scratchpad: Annotated[str, "scratchpad"]
    
class IndependentScratchpadState(TypedDict):
    print("accessing the state")
    messages:  Annotated[list, add_messages]
    personal_scratchpads:  Annotated[dict, operator.or_]

def create_select_random_agent_node(agent_names):
    def select_random_agent(state: IndependentScratchpadState)-> str:
        """
        This function returns a string of a random agent to start the conversation after having the personal scratchpads filled
        """
        agent_name = choice(agent_names)
        print(f"selected agent: {agent_name}")
        return agent_name
    return select_random_agent

def send_personal_scratchpad_calls(agents_info) -> List[Send]:
    def send_personal_scratchpad_prompt(state: IndependentScratchpadState):
        """
        This function triggers the personal scratchpad generation for each agent by sending a command to the personal_scratchpad_fill funcition
        """
        calls = []
        for agent_name, agent_type in agents_info.items():
            calls.append(Send(f"agent_scratchpad_fill", {"state": state, "agent_name" : agent_name, "agent_type" : agent_type}))
        return calls
    return send_personal_scratchpad_prompt
    
def create_personal_scratchpad_fill_node(model):
    def personal_scratchpad_fill(args): # Had to give it as dict because of the Send function
        agent_name = args["agent_name"]
        agent_type = args["agent_type"]
        state = args["state"]
        temp_prompt = []
        temp_prompt.append(SystemMessage(get_mbti_prompt(agent_type)))
        temp_prompt.extend(state["messages"])
        temp_prompt.append(SystemMessage("Add only your notes here to your personal scratchpad before interacting with the other agents:\n"))
        reply = model.invoke(temp_prompt)
        output = reply.content
        print(f"filled scratchpad of {agent_name} with: {output}")
        return Command(update={"personal_scratchpads": {agent_name: output}})
    return personal_scratchpad_fill

def create_judge_solution_node(model):
    def send_solution(state: IndependentScratchpadState):
        instruction = SystemMessage("You are an LLM as judge and you need to evaluate which of the solution options is most likely according to the task and what the last message was. Give your explanation in the *explanation* field and the solution that was given in *solution*, according to the options format.")
        task = HumanMessage(state["messages"][0].content)
        solution_options = eval(state["messages"][1].content.split("options:")[1].strip())
        print(solution_options)
        schema = TypedDict("structure", {"explanation": str, "solution": Literal[*solution_options]})
        response = model.with_structured_output(schema).invoke([instruction, task])
        explanation = response["explanation"]
        solution = response["solution"]
        return Command(update={"messages": [AIMessage(content=explanation, name = "judge"),AIMessage(content=solution, name="FINAL_ANSWER")]})
    return send_solution
 
def create_agent_network_node_function(agent_names, agent_name, agent_type, model):
    agent_names_execept_oneself = [string for string in agent_names if string != agent_name]
    structure = TypedDict("structure", {"next": Literal[*agent_names_execept_oneself, "send_solution"], "content": str}) # Change name judge solution to something more neutral
    scratchpad_prompt = SystemMessage(get_scratchpad_agent_prompt(agent_names), name="system")
    agents_personality_prompt = SystemMessage(get_mbti_prompt(agent_type), name="system")
    agent_scratchpad_reminder = SystemMessage(get_agent_scratchpad_reminder(), name="system")
    def agent(state: IndependentScratchpadState) -> Command[Literal[*agent_names_execept_oneself, "send_solution"]]:
        personal_scratchpad = AIMessage("#You Personal Scratchpad:\n" + state["personal_scratchpads"][agent_name], name=agent_name)
        temporary_messages = []
        temporary_messages = state["messages"].copy()
        temporary_messages.insert(0,"#Message History")
        temporary_messages.insert(0, agent_scratchpad_reminder)
        temporary_messages.insert(0, personal_scratchpad)
        temporary_messages.insert(0, scratchpad_prompt)
        temporary_messages.insert(0, agents_personality_prompt)
        response = model.with_structured_output(structure).invoke(temporary_messages)
        print(f"\n {agent_name} {agent_type}: '{response['content']}'")
        return Command(goto=response["next"],
                        update={"messages": AIMessage(content=response["content"], name=agent_name)}
                        )
    return agent

def solve_task(dataset_name, dataset_config, model_name, agent_types_strings, n_tasks = -1, runs = 1): 
    """
    Solve the given task.

    Args:
        model_name (str): Name of the language model.
        agent_types_strings (List[str]): List of the MBTI types for the agents.
    """
    model = get_model(model_name = model_name)
    agents_info = {f"agent{i}": agent_type for i, agent_type in enumerate(agent_types_strings)}
    
    # Add nodes to graph
    builder = StateGraph(IndependentScratchpadState)
    builder.add_node("agent_scratchpad_fill", create_personal_scratchpad_fill_node(model))
    for agent_name, agent_type in agents_info.items():
        builder.add_node(agent_name, create_agent_network_node_function(agent_names = agents_info.keys(), agent_name = agent_name, agent_type = agent_type, model = model))
    builder.add_node("send_solution", create_judge_solution_node(model))
    
    # Add edges
    agent_route_dictionary = {agent_name: agent_name for agent_name in agents_info.keys()}
    builder.add_conditional_edges(
        source = START,
        path = send_personal_scratchpad_calls(agents_info),
        path_map = ["agent_scratchpad_fill"]
    )
    builder.add_conditional_edges(
        source = 'agent_scratchpad_fill',
        path = create_select_random_agent_node(list(agents_info.keys())),
        path_map = agent_route_dictionary
    )
    builder.add_edge("send_solution", END)
    
    # Compile graph
    graph = builder.compile()
    print(graph.get_graph().draw_mermaid())
    # Invoke the graph
    print("Invoking the graph")
    invoke_graph_one_shot(graph, dataset_name, dataset_config, agent_types_strings, "blackboard_independent_scratchpad", n_tasks, runs)
