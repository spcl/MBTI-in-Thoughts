# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

# If you want to use this code, please update client_id and client_secret in lines 32 and 33.

import os
import pandas as pd

from TopUpvotedStoriesDataset import TopUpvotedStoriesDataset

# Initialize with the same parameters as the original class
top_stories_dataset = TopUpvotedStoriesDataset(
    source_path=r"dataset\train.wp_source",
    # We don't need target_path since we're just getting the prompts and finding top stories
)

# Check the current progress in the beginning
status = top_stories_dataset.get_processing_status()
print(f"Progress: {status['percentage']:.2f}% ({status['processed']}/{status['total']} items)")

# Create a directory for interim results if it doesn't exist
results_dir = r"results"
os.makedirs(results_dir, exist_ok=True)

# Fetch top stories
top_stories_dataset.fetch_top_stories_in_batches(
    client_id="",
    client_secret="",
    user_agent="python:primed_writers:v1.0 ",
    batch_size=500,  # Adjust based on how many items you can process in ~2 hours
    final_save_path=os.path.join(results_dir, "top_pre_2020_stories_final.csv")
)

# Check the current progress afterwards
status = top_stories_dataset.get_processing_status()
print(f"Progress: {status['percentage']:.2f}% ({status['processed']}/{status['total']} items)")

# Get engagement statistics 
try:
    stats = top_stories_dataset.get_engagement_stats()
    print("\nEngagement Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
except ValueError as e:
    print(f"Can't get statistics yet: {e}")

# Verify the data by loading previously saved batches
try:
    import glob
    batch_files = glob.glob("top_stories_batch_*.csv")
    if batch_files:
        print(f"\nFound {len(batch_files)} batch files. Loading a sample for verification:")
        sample_file = batch_files[-1]  # Use the most recent batch
        sample_data = pd.read_csv(sample_file)
        
        print(f"Batch file: {sample_file}")
        print(f"Number of entries: {len(sample_data)}")
        print(f"Columns: {sample_data.columns.tolist()}")
        
        # Check if we have stories
        has_stories = sample_data['top_story'].notna().sum()
        print(f"Entries with stories: {has_stories} ({has_stories/len(sample_data)*100:.2f}%)")
        
        # Display a sample
        if not sample_data.empty and has_stories > 0:
            print("\nSample story:")
            sample_row = sample_data[sample_data['top_story'].notna()].iloc[0]
            print(f"Prompt: {sample_row['story_prompt'][:100]}...")
            print(f"Author: {sample_row['story_author']}")
            print(f"Upvotes: {sample_row['story_upvotes']}")
            print(f"Story excerpt: {str(sample_row['top_story'])[:200]}...")
except Exception as e:
    print(f"Error verifying saved data: {e}")
