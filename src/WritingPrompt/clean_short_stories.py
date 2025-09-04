# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import os
import json
import argparse

def clean_short_stories(root_dir: str) -> None:
    """
    Deletes JSON files in root_dir if the 'model_output.story' field has less than 100 words.

    :param root_dir: Path to the directory, where the stories are stored.
    :type root_dir: str
    """
    deleted_files_count = 0
    total_files_checked = 0
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                total_files_checked += 1
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if 'model_output' in data and 'story' in data['model_output']:
                        story_text = data['model_output']['story']
                        if isinstance(story_text, str):
                            word_count = len(story_text.split())
                            
                            if word_count < 100:
                                print(f"Deleting {file_path} (word count: {word_count})")
                                os.remove(file_path)
                                deleted_files_count += 1
                        else:
                            print(f"Skipping {file_path}: 'story' in 'model_output' is not a string.")
                    else:
                        print(f"Skipping {file_path}: 'model_output' with 'story' key not found.")

                except json.JSONDecodeError:
                    print(f"Skipping {file_path}: Invalid JSON.")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"\nFinished cleaning.")
    print(f"Checked {total_files_checked} JSON files.")
    print(f"Deleted {deleted_files_count} files with less than 100 words in their 'story'.")


def count_words_in_stories(path: str) -> None:
    """
    Counts the words in the 'model_output.story' field of JSON files as well as printing some
    statistics. The path can be a single file or a directory.

    :param path: Path to the directory, where the stories are stored.
    :type path: str
    """
    total_word_count = 0
    total_files_checked = 0
    word_counts = []

    if os.path.isfile(path):
        files_to_check = [path]
        base_dir = os.path.dirname(path)
    elif os.path.isdir(path):
        files_to_check = []
        for subdir, _, files in os.walk(path):
            for file in files:
                if file.endswith('.json'):
                    files_to_check.append(os.path.join(subdir, file))
    else:
        print(f"Error: Path not found at '{path}'")
        return

    for file_path in files_to_check:
        total_files_checked += 1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'model_output' in data and isinstance(data['model_output'], dict) and 'story' in data['model_output']:
                story_text = data['model_output']['story']
                if isinstance(story_text, str):
                    word_count = len(story_text.split())
                    word_counts.append(word_count)
                    total_word_count += word_count
                    print(f"{os.path.basename(file_path)}: {word_count} words")
                else:
                    print(f"Skipping {os.path.basename(file_path)}: 'story' in 'model_output' is not a string.")
            else:
                print(f"Skipping {os.path.basename(file_path)}: 'model_output' with 'story' key not found.")

        except json.JSONDecodeError:
            print(f"Skipping {os.path.basename(file_path)}: Invalid JSON.")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    if total_files_checked > 1:
        average_word_count = total_word_count / len(word_counts) if word_counts else 0
        print(f"\n--- Summary ---")
        print(f"Checked {total_files_checked} JSON files.")
        print(f"Total words: {total_word_count}")
        print(f"Average words per story: {average_word_count:.2f}")
    elif not files_to_check:
        print("No JSON files found to count.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean up or count words in short stories from a file or directory.")
    parser.add_argument(
        '--action',
        type=str,
        default='clean',
        choices=['clean', 'count'],
        help="The action to perform: 'clean' to delete short stories, 'count' to get word counts."
    )
    parser.add_argument(
        '--path', 
        type=str, 
        default='written_stories',
        help='The path to a file or directory to process.'
    )
    args = parser.parse_args()

    target_path = args.path
    
    if not os.path.exists(target_path):
        print(f"Error: Path not found at '{target_path}'")
    else:
        if args.action == 'clean':
            if os.path.isdir(target_path):
                print(f"Starting to clean stories in '{target_path}'...")
                clean_short_stories(target_path)
            else:
                print("Error: The 'clean' action can only be used with a directory.")
        elif args.action == 'count':
            print(f"Starting to count words in stories in '{target_path}'...")
            count_words_in_stories(target_path) 
