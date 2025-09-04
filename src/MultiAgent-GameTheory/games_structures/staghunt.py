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

game_description = '''**You are a player in a repeated Stag Hunt game with communication.**  
## Instructions:
Each round, you and the other player must each choose to **hunt Stag** or **hunt Hare** without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.  
You have access to the game history.  
You are competing to win the most points in this game. Your goal is not necessarily fairness or cooperation, but to maximize your own reward.  
## Outcomes:
- **Both cooperate**: both players get 3 point.
- **One defects**: The defector gets 5 points, the cooperator gets 0.
- **Both DEFECT**: both players get 1 points.
## Notes:
- You are **not bound** to follow through on what you wrote. You can lie, bluff, or stay true to your word.  
- You will only see the other player's message **after you’ve sent yours**—so you can’t change your message based on theirs.  
- Once messages are exchanged, you must **choose your action (Hunt Stag or Hunt Hare)**, knowing that they may or may not follow their message.'''

class StagHuntActionResponse(BaseModel):
    """
    Respond with action to take: stag or hare.
    """
    action: Literal["stag", "hare"]

class StagHuntGame(BaseGameStructure):
    """
    Structured class for the Stag Hunt game.
    """
    @property
    def game_name(self) -> str:
        return "stag_hunt"
    
    @property
    def ActionResponse(self):
        return StagHuntActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_description)

    @property
    def payoff_matrix(self):
        return {
            ("stag", "stag"): (10, 10),
            ("stag", "hare"): (1, 8),
            ("hare", "stag"): (8, 1),
            ("hare", "hare"): (5, 5)
        }
