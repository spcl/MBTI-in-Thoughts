# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Mathis Lindner

import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Set all env vars to have the same name automatically
load_dotenv()
for k, v in os.environ.items():
    os.environ[k] = v

def get_model_by_id_and_provider(model_id: str, provider: str = None):
    properties = {
        "temperature": 0,
        "max_retries": 3
        # "timeout": 2000
    }
    if provider != "anthropic":
        properties["seed"] = 42  # Seed doesn’t work with Anthropic models
        
    if provider is None:
        return init_chat_model(
            model_id,
            **properties
        )
    return init_chat_model(
        model_id,
        model_provider=provider,
        **properties
    )
