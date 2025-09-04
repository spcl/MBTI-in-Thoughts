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

game_prompt_chicken = '''**You are a player in a repeated Chicken game with communication.**
## Instructions:
Each round, you and the other player must each choose to **swerve** or **stay** without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.  
You have access to the game history.  
You are competing to win the most points in this game. Your goal is not necessarily fairness or cooperation, but to maximize your own reward.  
## Outcomes:
- **Both stay**: both players crash and get 0 points.
- **One swerves, one stays**: The one who stays gets 10 points, and the one who swerves gets 1.
- **Both swerve**: both players get 0 points.  
## Notes:
- You are competing to win the most points in this game. Your goal is not to ensure fairness but to get the highest possible reward for yourself.
- You can change your strategy at any time.'''

class ChickenGameActionResponse(BaseModel):
    """
    Respond with action to take: swerve or stay.
    """
    action: Literal["swerve", "stay"]


class ChickenGame(BaseGameStructure):
    """
    Structured class for the Chicken game.
    """
    @property
    def game_name(self):
        return "chicken"
    @property
    def ActionResponse(self):
        return ChickenGameActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_chicken)

    @property
    def payoff_matrix(self):
        return {
            ("swerve", "swerve"): (0, 0),
            ("swerve", "stay"): (1, 10),
            ("stay", "swerve"): (10, 1),
            ("stay", "stay"): (0, 0)
        }
