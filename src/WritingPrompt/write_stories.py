# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import argparse
import json
import os
import time

from story_generation import write_story
from TopUpvotedStoriesLoaderDataset import TopUpvotedStoriesLoaderDataset
from tqdm import tqdm
from WritingPromptDataset import WritingPromptDataset

def save_story(path,
                story_prompt, 
                story_id,
                personality_type, 
                model_output, 
                usage_metadata, 
                temperature):
    """Save the story as a JSON file."""
    # Save story as JSON file
    contents = {
        "story_prompt": story_prompt,
        "story_id": story_id,
        "personality_type": personality_type,
        "model_output": model_output,
        "temperature": temperature,
        "usage_metadata": usage_metadata
    }
    with open(path, 'w') as f:
        json.dump(contents, f)


if __name__ == '__main__':
    personality_types = ['INTJ', 
                        'INTP', 
                        'INFP', 
                        'INFJ', 
                        'ISTJ', 
                        'ISTP', 
                        'ISFJ', 
                        'ISFP', 
                        'ENTJ', 
                        'ENTP', 
                        'ENFP', 
                        'ENFJ', 
                        'ESTJ', 
                        'ESTP', 
                        'ESFJ', 
                        'ESFP',
                        'EXPERT',
                        'NONE']
    # Load a personality type from general_priming.json
    # or use the default personality type
    with open('../../priming/priming_mbti.json', 'r') as f:
        personality_types_descriptions = json.load(f)

    # Load all contents of the file prompts.txt into a single string
    with open('prompts.txt', 'r') as f:
        prompt_template = f.read()

    temperature = 0
    count = 0
    parser = argparse.ArgumentParser(description='Generate stories for each personality type using story prompts from the WritingPrompts dataset')
    parser.add_argument('--number_of_stories', type=int, default=-1, help='Number of stories to generate')
    parser.add_argument('--generate_from', type=int, default=-1, help='Where to start generating from') # NOTE: this is different from the prompt index
    parser.add_argument('--generate_to', type=int, default=-1, help='Where to end generating at')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the models')
    parser.add_argument('--model_name', type=str, default='google/gemma-3n-e4b-it:free', help='Model from OpenRouter to use for generation. The model name may contain slashes.')
    parser.add_argument('--output_dir', type=str, default='debugs', help='Output directory for the generated stories')
    parser.add_argument('--dataset_type', type=str, default='top_upvoted', help='Type of dataset to use: writing_prompt or top_upvoted')
    parser.add_argument('--top_upvoted_csv', type=str, default='dataset/top_stories_batch_1.csv', help='Path to the CSV file containing top upvoted stories')
    args = parser.parse_args()

    number_of_stories = args.number_of_stories
    temperature = args.temperature
    generate_from = args.generate_from
    generate_to = args.generate_to
    model_name = args.model_name

    if args.dataset_type == 'writing_prompt':
        dataset = WritingPromptDataset(source_path = 'dataset/train.wp_source',
                                   target_path = 'dataset/train.wp_target',
                                   start_index = generate_from,
                                   end_index = generate_to,
                                   min_words = 25)
    elif args.dataset_type == 'top_upvoted':
        if args.top_upvoted_csv is None:
            raise ValueError("Must provide a CSV file path when using the top_upvoted dataset type")
        
        dataset = TopUpvotedStoriesLoaderDataset(csv_path=args.top_upvoted_csv,
                                              start_index=generate_from,
                                              end_index=generate_to,
                                              min_words=25)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    output_dir = ''.join(['written_stories/',args.output_dir])

    for i, data_item in tqdm(enumerate(dataset)):
        for personality_type in personality_types:
            story_prompt = data_item['story_prompt']
            story_id = data_item['id']
            story_object, story_json, prompt = write_story(story_prompt= story_prompt, 
                                        personality_type= personality_types_descriptions[personality_type], 
                                        prompt_template= prompt_template,
                                        temperature=temperature,
                                        model_name=model_name)
            # Add 1 second delay after each write_story call
            time.sleep(1)
            # Format i to be 6 digits long
            story_index_to_save_file = str(story_id).zfill(6)
            save_directory = ''.join([output_dir, '/',personality_type])
            # Create directory if it does not exist
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            save_story(path= ''.join([save_directory, '/', story_index_to_save_file, '_', personality_type, '.json']),
                    story_prompt= story_prompt, 
                    story_id= story_id,
                    personality_type= personality_type, 
                    model_output= story_json,
                    usage_metadata= story_object.usage_metadata,
                    temperature= temperature)
        count += 1
        if count >= number_of_stories and number_of_stories != -1:
            break
