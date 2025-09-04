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

load_dotenv()

def get_model(model_name: str, max_new_tokens: int=100):
    if model_name == "test":
        from langchain.chat_models import FakeChatModel
        return FakeChatModel(name="test", responses=["This is the random test response"])
    
    # If name starts with gpt..
    elif model_name.startswith("gpt"):
        from langchain_openai import ChatOpenAI
        logging.info(f"Creating OpenAI model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            seed=42,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    else: # Assuming it is a huggingface model
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from torch.cuda import is_available
        HF_HOME = os.getenv("HF_HOME") # Set cache directory
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
        device = 0 if is_available() else -1
        try:
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                device=device,        
                pipeline_kwargs=dict(
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.0,
                ),
            )            
            llm.pipeline.tokenizer.pad_token = llm.pipeline.tokenizer.eos_token
            hf_chat = ChatHuggingFace(llm = llm)
        except Exception as e:
            raise(f"Error in creating HuggingFace model: {e}")
        return hf_chat
