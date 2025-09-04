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

# Game: Deadlock
game_prompt_deadlock = '''**You are a player trapped in a repeated Deadlock game with communication.**
## Instructions:
Each round, you and the other player must each choose to **Cooperate** or **Defect**, without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.
- You have access to the game history above.
- You will only see the other player's message after you've sent yours, same for the action.
- Once messages are exchanged, you must choose whether to cooperate or defect.

## Outcomes:
- **Both defect**: both players get 2 points.
- **One defects, one cooperates**: the defector gets 3 points, the cooperator gets 0.
- **Both cooperate**: both players get 1 point.

## Notes:
- You are competing to win the most points in this game. Your goal is not to encourage cooperation, but to maximize your own reward.
- You can change your strategy at any time.'''

class DeadlockActionResponse(BaseModel):
    """
    Respond with action to take: cooperate or defect.
    """
    action: Literal["cooperate", "defect"]


class DeadlockGame(BaseGameStructure):
    """
    Structured class for the Deadlock game.
    """
    @property
    def game_name(self):
        return "deadlock"

    @property
    def ActionResponse(self):
        return DeadlockActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_deadlock)

    @property
    def payoff_matrix(self):
        return {
            ("cooperate", "cooperate"): (1, 1),
            ("cooperate", "defect"): (0, 3),
            ("defect", "cooperate"): (3, 0),
            ("defect", "defect"): (2, 2)
        }
