# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import argparse
from collections import Counter
import json
import matplotlib.pyplot as plt
import nltk
import os
import pandas as pd
import textwrap
from tqdm import tqdm
from typing import Dict, List, Tuple

from evaluators import LexicalRichnessEvaluator, LMEvaluator, ReadabilityEvaluator
from plot import plot_llm_evaluations, plot_metrics
from TopUpvotedStoriesLoaderDataset import TopUpvotedStoriesLoaderDataset
from WritingPromptDataset import WritingPromptDataset


def evaluate_stories(
    stories: Dict,
    writer: str,
    evaluator: Evaluator,
    evaluation_save_path: str,
    from_idx: int = 0,
    to_idx: int = -1,
    num_stories: int = -1
) -> Dict:
    """
    Evaluate the stories with the chosen evaluator and write the results to files.

    :param stories: Stories to evaluate.
    :type stories: Dict
    :param writer: String to identify who wrote the story, e.g. a personality type.
    :type writer: str
    :param evaluator: Evaluator to use.
    :type evaluator: Evaluator
    :param evaluation_save_path: Path, where the results of the evaluation are stored.
    :type evaluation_save_path: str
    :param from_idx: Start index for the stories. Defaults to 0.
    :type from_idx: int
    :param to_idx: End index for the stories. Defaults to -1, indicating that all remaining stories are evaluated.
    :type to_idx: int
    :param num_stories: Maximum number of stories, that will be evaluated. Defaults to -1, indicating no limit.
    :type num_stories: int
    :return: Averaged metrics.
    :rtype: Dict
    """
    if evaluator.is_numerical:
        metric_sums = {key: 0 for key in evaluator.get_metric_names()}
    else:
        metric_sums = {key: [] for key in evaluator.get_metric_names()}

    evaluated_stories = 0
    evaluator_name = evaluator.__class__.__name__

    # Create evaluation save dir if it does not exist
    if not os.path.exists(f'{evaluation_save_path}/{evaluator_name}/{writer}'):
        os.makedirs(f'{evaluation_save_path}/{evaluator_name}/{writer}')

    # Convert to list and slice by index range
    story_items = list(stories.items())
    if to_idx == -1:
        to_idx = len(story_items)
    story_items = story_items[from_idx:to_idx]
    
    for story_id, story in tqdm(story_items, desc=f'Evaluating {evaluator_name}', total=len(story_items)):
        metrics = evaluator.evaluate(story)
        if metrics is None:
            print(f'Skipping story {story_id} because it could not be evaluated.')
            continue
            
        if evaluator.is_numerical:
            for key, value in metrics.items():
                metric_sums[key] += value
        else:
            for key, value in metrics.items():
                metric_sums[key].append(value)
        # Write the evaluation to a file
        with open(f'{evaluation_save_path}/{evaluator_name}/{writer}/{story_id}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        evaluated_stories += 1
        if evaluated_stories >= num_stories and num_stories != -1:
            break
    
    if evaluator.is_numerical:
        average_metrics = {key: value / len(story_items) if len(story_items) > 0 else 0 for key, value in metric_sums.items()}
    else:
        average_metrics = metric_sums
    return average_metrics


def valid_story_file_structure(data: Dict, item_name: str) -> bool:
    """
    Validate the data structure.

    :param data: Data to validate.
    :type data: Dict
    :param item_name: Name of the data item.
    :type item_name: str
    :return: True if the data structure is valid, False otherwise.
    :rtype: bool
    """
    if 'story_id' not in data.keys():
        print('story_id not found for:', item_name)
        return False
    if 'model_output' not in data or data['model_output'] is None:
        print('model_output not found or is None for:', item_name)
        return False
    if not isinstance(data['model_output'], dict):
        try:
            data['model_output'] = json.loads(data['model_output'])
        except (json.JSONDecodeError, TypeError):
            print('error loading data[\'model_output\'] for: ', item_name)
            return False
    try:
        story = data['model_output']['story']
    except KeyError:
        print('story not found for: ', item_name)
        return False
    return True


def enough_words_in_story(text: str) -> bool:
    """
    Check if the story has at least 100 words.

    :param text: Text to check.
    :type text: str
    :return: True, if the text contains at least a 100 words, False otherwise.
    :rtype: bool
    """
    return len(text.split()) >= 100

    
def evaluate(
    all_stories: Dict,
    metrics_to_evaluate: List[str],
    evaluation_save_path: str,
    from_idx: int = 0,
    to_idx: int = -1,
    num_stories: int = -1,
    eval_model_name: str = 'Qwen/Qwen2.5-14B-Instruct',
    binary_evaluation: bool = False
) -> Tuple[Dict, Dict]:
    """
    Evaluate the stories (in a range) with the supplied metrics.

    :param all_stories: Stories to evaluate.
    :type all_stories: Dict
    :param metrics_to_evaluate: Metrics to use for the evaluation of the stories.
    :type metrics_to_evaluate: List[str]
    :param evaluation_save_path: Path, where the results of the evaluation are stored.
    :type evaluation_save_path: str
    :param from_idx: Start index for the stories. Defaults to 0.
    :type from_idx: int
    :param to_idx: End index for the stories. Defaults to -1, indicating that all remaining stories are evaluated.
    :type to_idx: int
    :param num_stories: Maximum number of stories, that will be evaluated. Defaults to -1, indicating no limit.
    :type num_stories: int
    :param eval_model_name: Name of the model to be used for LLM evaluation, if selected as a metric. Defaults to "Qwen/Qwen2.5-14B-Instruct".
    :type eval_model_name: str
    :param binary_evaluation: Flag to indicate whether binary evaluation should be employed. Defaults to False.
    :type binary_evaluation: bool
    :return: Tuple of the averaged metrics for each story as well as the evaluators.
    :rtype: Tuple[Dict, Dict]
    """
    metrics = {metric: {} for metric in metrics_to_evaluate}
    evaluators = {}
    for metric in metrics_to_evaluate:
        if metric == 'LLM':
            evaluators[metric] = LLMEvaluator(model_name=eval_model_name, binary_evaluation=binary_evaluation)
        else:
            evaluators[metric] = globals()[metric + 'Evaluator']()
    
    # Metrics could be LexicalRichness, Readability, LLM, etc.
    for metric in metrics_to_evaluate:
        print(' -------- EVALUATING METRIC:', metric, '--------')
        # Use metric name to get the correct evaluator
        for ptype, stories in tqdm(all_stories.items(), desc=f'Processing '):
            print('PTYPE:', ptype)
            if len(stories) < num_stories and num_stories != -1:
                print(f'Not enough stories for {ptype}')
                continue
            # Evaluate all stories for a particular personality type
            average_metrics = evaluate_stories(stories = stories, 
                                               writer = ptype, 
                                               evaluator = evaluators[metric],
                                               evaluation_save_path = evaluation_save_path,
                                               from_idx = from_idx,
                                               to_idx = to_idx,
                                               num_stories = num_stories)
            metrics[metric][ptype] = average_metrics
    return metrics, evaluators


def collect_stories(ptypes: List[str], load_path: str) -> Tuple[Dict, List[int]]:
    """
    Collect and return the stories written by the various personality types.

    :param ptypes: List of personality types.
    :type ptypes: List[str]
    :param load_path: Path to the directory from where the stories are collected.
    :type load_path: str
    :return: Tuple of stories as well as a list of story IDs for which the stories were written.
    :rtype: Tuple[Dict, List[int]]
    """
    story_ids = []
    ptype_stories = {ptype: {} for ptype in ptypes}

    for ptype in ptypes:
        # Load all json files in directories named after the personality types
        files = [f for f in os.listdir(os.path.join(load_path, ptype)) if f.endswith('.json')]
        # ptype_stories collects all the stories a particular personality type has written
        for item in files:
            # Load json file
            with open(os.path.join(load_path, ptype, item), 'r') as f:
                data = json.load(f)
                # Check if the file has the correct structure
                if not valid_story_file_structure(data, item):
                    continue

                story = data['model_output']['story']

                if not enough_words_in_story(story):
                    continue

                # Keep story id to later match with human generated stories to check against human performance
                story_id = data['story_id']
                story_ids.append(story_id)
                ptype_stories[ptype][story_id] = story
    
    story_ids = list(set(story_ids))
    return ptype_stories, story_ids


def collect_human_written_stories(story_ids: List[int], target_dataset: WritingPromptDataset) -> Dict:
    """
    Collect and return the stories written previously by humans.

    :param story_ids: List of story IDs.
    :type story_ids: List[int]
    :param target_dataset: Dataset of stories.
    :type target_dataset: WritingPromptDataset
    :return: Collected stories.
    :rtype: Dict
    """
    human_stories = {}
    total_human_stories = 0
    for story in target_dataset:
        if story['id'] in story_ids:
            total_human_stories += 1
            if enough_words_in_story(story['human_story']):
                human_stories[story['id']] = story['human_story']
            else:
                print(f'Skipping human story {story["id"]} because it has less than 100 words')

    print(f'Total human stories found: {total_human_stories}')
    print(f'Human stories with >= 100 words: {len(human_stories)}')
    return human_stories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate stories generated by the models primed with personality type descriptions')
    parser.add_argument('--load_path', type=str, default='qwen3_235B_A22B/round_1', help='Load path for the generated stories')
    parser.add_argument('--plot_save_path', type=str, default='plots_debugs', help='Where to save the plots')
    parser.add_argument('--evaluation_save_path', type=str, default='evaluations_debugs', help='Where to save the evaluations')
    parser.add_argument('--from_idx', type=int, default=0, help='Start index for stories to evaluate')
    parser.add_argument('--to_idx', type=int, default=-1, help='End index for stories to evaluate (-1 for all)')
    parser.add_argument('--dataset_type', type=str, default='top_upvoted', help='Type of dataset to use: writing_prompt or top_upvoted')
    parser.add_argument('--top_upvoted_csv', type=str, default='top_stories_batch_1.csv', help='Path to the CSV file containing top upvoted stories')
    parser.add_argument('--eval_model_name', type=str, default='Qwen/Qwen2.5-14B-Instruct', help='Model to use for evaluation')
    parser.add_argument('--binary_evaluation', type=str, default='True', help='Use binary evaluation for LLM evaluator. Pass "true" or "false".')
    args = parser.parse_args()
   
    # Initialize the appropriate dataset based on the type
    if args.dataset_type == 'writing_prompt':
        target_dataset = WritingPromptDataset(
            source_path='dataset/train.wp_source',
            target_path='dataset/train.wp_target',
            min_words=25
        )
    elif args.dataset_type == 'top_upvoted':
        if args.top_upvoted_csv is None:
            raise ValueError("Must provide a CSV file path when using the top_upvoted dataset type")
        top_upvoted_csv_path = ''.join(['dataset/', args.top_upvoted_csv])
        target_dataset = TopUpvotedStoriesLoaderDataset(
            csv_path=top_upvoted_csv_path,
            min_words=25
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    load_path_base = 'written_stories/'
    load_path = ''.join([load_path_base, args.load_path])
    plot_save_path = args.plot_save_path
    evaluation_save_path = ''.join(['evaluations/', args.evaluation_save_path])
    # List of all directories in the load path
    ptypes = [f for f in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, f))]
    # Sort ptypes by name
    ptypes.sort()

    ptype_stories, story_ids = collect_stories(ptypes, load_path)
    human_stories = collect_human_written_stories(story_ids, target_dataset)
    # Add human_stories to ptype_stories under the key 'human' and keep in a new dictionary called 'all_stories'.
    # This is solely to improve code readability.
    ptype_stories['human'] = human_stories
    all_stories = ptype_stories

    metrics_to_evaluate = [
                            'Readability', 
                            # 'LexicalRichness', 
                            # 'LLM'
                            ]
    metrics, evaluators = evaluate(all_stories = all_stories, 
                                   metrics_to_evaluate = metrics_to_evaluate,
                                   evaluation_save_path = evaluation_save_path,
                                   from_idx = args.from_idx,
                                   to_idx = args.to_idx,
                                   num_stories = -1,
                                   eval_model_name = args.eval_model_name,
                                   binary_evaluation = args.binary_evaluation.lower() == 'true')
