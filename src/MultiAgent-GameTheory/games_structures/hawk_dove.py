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

# Game: Hawk-Dove
game_prompt_hawk_dove = '''**You are a player trapped in a repeated Hawk-Dove game with communication.**
## Instructions:
Each round, you and the other player must each choose between playing **Hawk** (aggressive) or **Dove** (peaceful), without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.
- You have access to the game history above.
- You will only see the other player's message after you've sent yours, same for the action.
- Once messages are exchanged, you must choose whether to play Hawk or Dove.

## Outcomes:
- **Both choose Hawk**: both players get 0 points due to conflict.
- **One chooses Hawk, one chooses Dove**: the Hawk gets 3 points, the Dove gets 1 point.
- **Both choose Dove**: both players get 2 points.

## Notes:
- You are competing to win the most points in this game. Your goal is not to maintain peace or fairness, but to get the highest possible reward for yourself.
- You can change your strategy at any time.'''

class HawkDoveActionResponse(BaseModel):
    """
    Respond with action to take: hawk or dove.
    """
    action: Literal["hawk", "dove"]


class HawkDoveGame(BaseGameStructure):
    """
    Structured class for the Hawk-Dove game.
    """
    @property
    def game_name(self):
        return "hawk_dove"

    @property
    def ActionResponse(self):
        return HawkDoveActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_hawk_dove)

    @property
    def payoff_matrix(self):
        return {
            ("hawk", "hawk"): (0, 0),
            ("hawk", "dove"): (3, 1),
            ("dove", "hawk"): (1, 3),
            ("dove", "dove"): (2, 2)
        }
