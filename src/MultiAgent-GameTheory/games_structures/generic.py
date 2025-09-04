# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

from games_structures.base_game import BaseGameStructure
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Literal

game_prompt_generic = '''**You are a player in a repeated game with communication.**
## Instructions:
Each round, you and the other player must each choose to **cooperate** or **defect** without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.
You have access to the game history.
You are competing to win the most points in this game. Your goal is not necessarily fairness or cooperation, but to maximize your own reward.
## Outcomes:
- **Both cooperate**: both players get 3 points.
- **One defects**: The defector gets 5 points, the cooperator gets 0.
- **Both DEFECT**: both players get 1 point.
## Notes:
- You are competing to win the most points in this game. Your goal is not to ensure fairness or cooperation but to get the highest possible reward for yourself.
- You can change your strategy at any time'''

class GenericActionResponse(BaseModel):
    """
    Respond with action to take: cooperate or defect.
    """
    action: Literal["cooperate", "defect"]


class GenericGame(BaseGameStructure):
    """
    Structured class for a generic repeated game with communication.
    """
    @property
    def game_name(self):
        return "generic"
    
    @property
    def ActionResponse(self):
        return GenericActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_generic)

    @property
    def payoff_matrix(self):
        return {
            ("cooperate", "cooperate"): (3, 3),
            ("cooperate", "defect"): (0, 5),
            ("defect", "cooperate"): (5, 0),
            ("defect", "defect"): (1, 1)
        }
