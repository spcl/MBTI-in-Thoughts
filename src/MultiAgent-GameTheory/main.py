# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import argparse
import json

from run_games_mbti import run_n_rounds_w_com
from datetime import datetime

def main(args):
    date_string = datetime.now().strftime("%y%m%d")
    output_dir = "data/outputs/"
    base_game_state_path = f"{date_string}"
    game_state_path = output_dir + f"{base_game_state_path}.csv"
    game_state = run_n_rounds_w_com(
        model_name_1=args.model_id_1, 
        model_name_2=args.model_id_2, 
        model_provider_1=args.model_provider_1 if args.model_provider_1 else None,
        model_provider_2=args.model_provider_2 if args.model_provider_2 else None,
        total_rounds=args.rounds, 
        personality_key_1=args.agent_1_persona, 
        personality_key_2=args.agent_2_persona, 
        game_name=args.game_name, 
        file_path=game_state_path
    )

if __name__ == "__main__":
    personality_choices = json.load(open("../../prompting/priming_without_mention_of_mbti_different_none_with_altruistic_selfish.json")).keys()
    game_names = ["prisoners_dilemma", "stag_hunt", "generic", "chicken", "coordination", "hawk_dove", "deadlock", "battle_of_sexes"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id_1", type=str, help="The first model id to use for the game", required=True)
    parser.add_argument("--model_id_2", type=str, help="The second model id to use for the game", required=True)
    parser.add_argument("--model_provider_1", type=str, help="The provider of the first model", required=False)
    parser.add_argument("--model_provider_2", type=str, help="The provider of the second model", required=False)
    parser.add_argument("--rounds", type=int, help="The number of rounds to play", required=True)
    parser.add_argument("--agent_1_persona", choices = personality_choices, help="The personality of agent 1", required=True)
    parser.add_argument("--agent_2_persona", choices = personality_choices, help="The personality of agent 2", required=True)
    parser.add_argument("--game_name", choices = game_names, help="The game to play")
    args = parser.parse_args()
    main(args)
