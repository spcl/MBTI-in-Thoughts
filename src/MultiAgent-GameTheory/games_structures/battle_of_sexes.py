# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

from games_structures.base_game import BaseGameStructure
from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from typing import Literal

# Game: Battle of the Sexes
game_prompt_battle_of_sexes = '''**You are a player trapped in a repeated Battle of the Sexes game with communication.**
## Instructions:
Each round, you and the other player must each choose between attending **Football** or **Ballet**, without knowing what the other will choose. However, before making your decision, you can send a **message** to the other player.
- You have access to the game history above.
- You will only see the other player’s message after you’ve sent yours, same for the action.
- Once messages are exchanged, you must choose between Football and Ballet.

## Outcomes:
- **Both choose Football**: Player 1 gets 2 points, Player 2 gets 1.
- **Both choose Ballet**: Player 1 gets 1 point, Player 2 gets 2.
- **Different choices**: both players get 0 points.

## Notes:
- You are competing to win the most points in this game. Your goal is not to ensure coordination or fairness, but to get the highest possible reward for yourself.
- You can change your strategy at any time.'''

class BattleOfSexesActionResponse(BaseModel):
    """
    Respond with action to take: football or ballet.
    """
    action: Literal["football", "ballet"]


class BattleOfSexesGame(BaseGameStructure):
    """
    Structured class for the Battle of the Sexes game.
    """
    @property
    def game_name(self):
        return "battle_of_sexes"

    @property
    def ActionResponse(self):
        return BattleOfSexesActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_battle_of_sexes)

    @property
    def payoff_matrix(self):
        return {
            ("football", "football"): (2, 1),
            ("football", "ballet"): (0, 0),
            ("ballet", "football"): (0, 0),
            ("ballet", "ballet"): (1, 2)
        }
