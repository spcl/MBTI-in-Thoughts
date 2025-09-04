# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import pandas as pd

from games_structures.base_game import BaseGameStructure, GameState
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from models import get_model_by_id_and_provider
from node_helpers import load_game_structure_from_registry, get_answer_format, get_question_prompt, get_agent_annotated_prompt, AnnotatedPrompt
from typing import Literal, Callable

def send_prompts_node(prompt_type : Literal["message", "action"], GameStructure: BaseGameStructure) -> Callable:
    """
    Get the function to send the prompts to the agents. The function is used in the graph to send the prompts to the agents.

    Args:
        prompt_type (Literal["message", "action"]): The type of prompt to generate
        GameStructure (BaseGameStructure): The game structure to use
    Returns:
        Callable: The function to send the prompts to the agents
    """
    def send_prompts(state: GameState) -> list[Send]:
        """
        Send the prompts to the agents. The function is used in the graph to send the prompts to the agents.

        Args:
            state (GameState): The state of the game
        Returns:
            list[Send]: The list of sends to the agents
        """
        agent_1_annotated_prompt_state = get_agent_annotated_prompt("agent_1", state, prompt_type, GameStructure)
        agent_2_annotated_prompt_state = get_agent_annotated_prompt("agent_2", state, prompt_type, GameStructure)
        return [Send(f"invoke_from_prompt_state_{prompt_type}", agent_1_annotated_prompt_state), Send(f"invoke_from_prompt_state_{prompt_type}", agent_2_annotated_prompt_state)]
    return send_prompts


def invoke_from_prompt_state_node(models, GameStructure) -> Callable:
    """
    Get the function to invoke the model from the prompt state. The function is used in the graph to invoke the model from the prompt state.

    Args:
        models (dict): The models to use
        GameStructure (BaseGameStructure): The game structure to use
    Returns:
        Callable: The function to invoke the model from the prompt state
    """
    def invoke_from_prompt_state(state : AnnotatedPrompt) -> Command:
        """
        Invoke the model from the prompt state. The function is used in the graph to invoke the model from the prompt state.

        Args:
            state (AnnotatedPrompt): The prompt state
        Returns:
            Command: The updates for the state
        """
        json_mode = False
        try:
            for model in models.values():
                if model.model_name == "deepseek-chat":
                    json_mode = True
        except:
            pass

        prompt = state.prompt
        agent_name = state.agent_name
        prompt_type = state.prompt_type
        model = models[agent_name]
        Structure = GameStructure.MessageResponse if prompt_type == "message" else GameStructure.ActionResponse
        if json_mode:
            response = model.with_structured_output(Structure, method="json_mode", include_raw=True).invoke(prompt)
            message = ""
            if prompt_type == "message":
                message = response["parsed"].message
            else:
                message = response["parsed"].action
        else:
            response = model.with_structured_output(Structure).invoke(prompt)
            message = response.message if prompt_type == "message" else response.action
        print (f"Agent {agent_name} {prompt_type} : {message}")
        return Command(update = {f"{agent_name}_{prompt_type}s": [message]})
    return invoke_from_prompt_state


def judge_intent_node(model, GameStructure) -> Callable: # Use OpenAI GPT4o-mini for judging
    """
    Get the function to judge the intent of the agents. The function is used in the graph to judge the intent of the agents.

    Args:
        model (str): The model to use
        GameStructure (BaseGameStructure): The game structure to use
    Returns:
        Callable: The function to judge the intent of the agents
    """
    def judge_intent(state: GameState) -> Command:
        """
        Judge the intent of the agents based on their messages and actions. The
        function is used in the graph to judge the intent of the agents.

        Args:
            state (GameState): The state of the game
        Returns:
            Command: The updates for the state
        """
        message_1 = state["agent_1_messages"][-1]
        message_2 = state["agent_2_messages"][-1]
        action_1 = state["agent_1_actions"][-1]
        action_2 = state["agent_2_actions"][-1]
        
        question = get_question_prompt(GameStructure)
        answer_format = get_answer_format(GameStructure)
        
        response_1 = model.with_structured_output(answer_format).invoke(
            f"{question} : {message_1}"
        )
        response_2 = model.with_structured_output(answer_format).invoke(
            f"{question} : {message_2}"
        )

        intent_agent_1 = response_1.answer
        intent_agent_2 = response_2.answer
        truthful_agent_1 = intent_agent_1 == action_1
        truthful_agent_2 = intent_agent_2 == action_2
        analysis_agent_1 = response_1.analysis
        analysis_agent_2 = response_2.analysis
        return Command(update = {
            "intent_agent_1": [intent_agent_1],
            "intent_agent_2": [intent_agent_2],
            "truthful_agent_1": [truthful_agent_1],
            "truthful_agent_2": [truthful_agent_2],
            "analysis_agent_1": [analysis_agent_1],
            "analysis_agent_2": [analysis_agent_2]
        })
    return judge_intent
        

def update_state_node(GameStructure):
    """
    Get the function to update the state of the game. The function is used in the graph to update the state of the game.

    Args:
        GameStructure (BaseGameStructure): The game structure to use
    Returns:
        Callable: The function to update the state of the game
    """
    def update_state(state: GameState) -> Command:
        """
        Update the state of the game based on the actions taken by the agents. The function is used in the graph to update the state of the game.

        Args:
            state (GameState): The state of the game
        Returns:
            Command: The updates for the state
        """
        state_updates = {}
        # Update scores
        agent_1_decision = state["agent_1_actions"][-1]
        agent_2_decision = state["agent_2_actions"][-1]
        score_agent1, score_agent2 = GameStructure.payoff_matrix[(agent_1_decision, agent_2_decision)]
        
        # Add scores to scores
        state_updates["agent_1_scores"] = [score_agent1]
        state_updates["agent_2_scores"] = [score_agent2]
        
        # Increment round
        state_updates["current_round"] = state["current_round"] + 1 
        return Command(update = state_updates)
    return update_state


def should_continue(state: GameState) -> bool:
    return (state["current_round"] <= state["total_rounds"])


def run_n_rounds_w_com(model_provider_1: str, model_name_1: str, model_provider_2: str, model_name_2: str, total_rounds: int, personality_key_1: str, personality_key_2: str, game_name: str, file_path: str) -> GameState:
    """
    Run the game for n rounds. The function is used to run the game for n rounds.

    Args:
        model_provider_1 (str): The provider of the first model
        model_name_1 (str): The name of the first model
        model_provider_2 (str): The provider of the second model
        model_name_2 (str): The name of the second model
        total_rounds (int): The number of rounds to run
        personality_key_1 (str): The personality key for the first agent
        personality_key_2 (str): The personality key for the second agent
        game_name (str): The name of the game to run
        file_path (str): The path to save the results
    Returns:
        GameState: The final state of the game
    """
    # Get models
    models = {
        "agent_1": get_model_by_id_and_provider(model_name_1, provider=model_provider_1),
        "agent_2": get_model_by_id_and_provider(model_name_2, provider=model_provider_2)
    }
    
    intent_model = get_model_by_id_and_provider("gpt-4o-mini")
    callback_handler = OpenAICallbackHandler()
    
    GameStructure = load_game_structure_from_registry(game_name)# Game now includes the game prompt, the payoff matrix, the message response the action response formats
    
    # Create graph
    graph = StateGraph(GameState, input = GameState, output = GameState)
    # Add nodes
    graph.add_node("lambda_to_messages", lambda x: {})
    graph.add_node("lambda_from_messages", lambda x: {})
    graph.add_node(f"invoke_from_prompt_state_message", invoke_from_prompt_state_node(models, GameStructure))
    graph.add_node(f"invoke_from_prompt_state_action", invoke_from_prompt_state_node(models, GameStructure))
    graph.add_node("judge_intent", judge_intent_node(intent_model, GameStructure))
    graph.add_node("update_state", update_state_node(GameStructure))
    
    # Add edges
    graph.add_edge(START, "lambda_to_messages")
    graph.add_conditional_edges(
        source = "lambda_to_messages", 
        path = send_prompts_node("message", GameStructure),
        path_map = ["invoke_from_prompt_state_message"]
        )
    graph.add_edge("invoke_from_prompt_state_message","lambda_from_messages")
    graph.add_conditional_edges(
        source = "lambda_from_messages", 
        path = send_prompts_node("action", GameStructure),
        path_map = ["invoke_from_prompt_state_action"]
        )
    graph.add_edge("invoke_from_prompt_state_action","judge_intent")
    graph.add_edge("judge_intent", "update_state")
    graph.add_conditional_edges(
        source = "update_state",
        path = should_continue,
        path_map = {
            False : END,
            True : "lambda_to_messages"
            }
        )
    # Compile and run
    compiled_graph = graph.compile()
    # Create initial state
    initial_state = GameState(
        personality_key_1=personality_key_1,
        personality_key_2=personality_key_2,
        current_round=1,
        total_rounds=total_rounds
    )
    end_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 200, "callbacks": [callback_handler]})
    print(f"Total Cost (USD): ${callback_handler.total_cost}")
    # Save results in pd df
    path_to_csv = file_path
    columns = ["model_provider_1", "model_name_1", "model_provider_2", "model_name_2", "personality_1", "personality_2", "agent_1_scores", "agent_2_scores", "agent_1_messages", "agent_2_messages", "agent_1_actions", "agent_2_actions", "intent_agent_1", "intent_agent_2", "truthful_agent_1", "truthful_agent_2", "analysis_agent_1", "analysis_agent_2", "total_rounds", "total_tokens", "total_cost_USD"]

    end_state["agent_1_messages"] = [msg.replace('"', "'") for msg in end_state["agent_1_messages"]]
    end_state["agent_2_messages"] = [msg.replace('"', "'") for msg in end_state["agent_2_messages"]]
    end_state["agent_1_actions"] = [action.replace('"', "'") for action in end_state["agent_1_actions"]]
    end_state["agent_2_actions"] = [action.replace('"', "'") for action in end_state["agent_2_actions"]]

    new_row = pd.DataFrame([{
        "game_name": game_name,
        "model_provider_1": model_provider_1,
        "model_name_1": model_name_1,
        "model_provider_2": model_provider_2,
        "model_name_2": model_name_2,
        "personality_1": personality_key_1,
        "personality_2": personality_key_2,
        "agent_1_scores": end_state["agent_1_scores"],
        "agent_2_scores": end_state["agent_2_scores"],
        "agent_1_messages": end_state["agent_1_messages"],
        "agent_2_messages": end_state["agent_2_messages"],
        "agent_1_actions": end_state["agent_1_actions"],
        "agent_2_actions": end_state["agent_2_actions"],
        "intent_agent_1": end_state["intent_agent_1"],
        "intent_agent_2": end_state["intent_agent_2"],
        "truthful_agent_1": end_state["truthful_agent_1"],
        "truthful_agent_2": end_state["truthful_agent_2"],
        "analysis_agent_1": end_state["analysis_agent_1"],
        "analysis_agent_2": end_state["analysis_agent_2"],
        "total_rounds": total_rounds,
        "total_tokens": callback_handler.total_tokens,
        "total_cost_USD": callback_handler.total_cost
    }])
    try:
        df = pd.read_csv(path_to_csv)
    except FileNotFoundError:
        df = new_row
    else:
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path_to_csv, mode='w', header=True, index=False)
    return end_state
