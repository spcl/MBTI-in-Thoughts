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

# Game: Coordination
game_prompt_coordination = '''**You are a player trapped in a repeated Coordination game with communication.**
## Instructions:
Each round, you and the other player must each choose between **Option A** and **Option B**, without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.
- You have access to the game history above.
- You will only see the other player’s message after you’ve sent yours, same for the action.
- Once messages are exchanged, you must choose between Option A and Option B.

## Outcomes:
- **Both choose A**: both players get 2 points.
- **Both choose B**: both players get 1 point.
- **Different choices**: both players get 0 points.

## Notes:
- You are competing to win the most points in this game. Your goal is not simply to coordinate, but to maximize your own reward.
- You can change your strategy at any time.'''

class CoordinationActionResponse(BaseModel):
    """
    Respond with action to take: A or B.
    """
    action: Literal["A", "B"]

class CoordinationGame(BaseGameStructure):
    """
    Structured class for the Coordination game.
    """
    @property
    def game_name(self):
        return "coordination"

    @property
    def ActionResponse(self):
        return CoordinationActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_coordination)

    @property
    def payoff_matrix(self):
        return {
            ("A", "A"): (2, 2),
            ("A", "B"): (0, 0),
            ("B", "A"): (0, 0),
            ("B", "B"): (1, 1)
        }
