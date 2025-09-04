# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import json
import os

from langchain-core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain-core.output_parsers import JsonOutputParser
from langchain-openai import OpenAI
from langchain-openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Story(BaseModel):
    """Story to be written."""
    relation_to_personality: str = Field(description="Describe how this task relates to your personality type. Indicate how your strengths and weaknesses are relevant to the task. Start your reasoning with the phrase \"I am a(n) [your personality type] and\". ")
    reasoning_related_to_personality: str = Field(description="Then, write a story using reasoning. During reasoning, link the reasoning steps explicitly to your personality. It is absolutely necessary that you reference traits of your personality during that reasoning.")
    story: str = Field(description="Write a story given the story prompt. Include creative elements, creative vocabulary and a plot.")


def evaluate_json_output(output, fixing_parser):
    """Evaluate the output of the model."""
    # Check if the output is a dictionary
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        print('Output is not a valid JSON!')
        output = fixing_parser.parse(output)
    return output


def write_story(story_prompt, personality_type, prompt_template, model_name: str = 'qwen/qwen3-235b-a22b:free', temperature: float = 0.0, max_tokens: int = -1):
    llm = ChatOpenAI(
        temperature=temperature,    
        model_name=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url='https://openrouter.ai/api/v1',
        max_tokens=2000,
    )

    parser = JsonOutputParser(pydantic_object = Story)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm, max_retries = 3)
    
    system_template = personality_type
    human_template = prompt_template

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())  

    prompt_and_model = prompt | llm

    story_written = prompt_and_model.invoke({"personality_type": personality_type, "story_prompt": story_prompt})
    story_written_json = evaluate_json_output(story_written.content, fixing_parser)
   
    return story_written, story_written_json, prompt
