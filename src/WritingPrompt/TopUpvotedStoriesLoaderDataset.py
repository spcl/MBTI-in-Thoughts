# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import matplotlib.pyplot as plt
import pandas as pd

class TopUpvotedStoriesLoaderDataset:
    """
    A class to load and process datasets created by TopUpvotedStoriesDataset
    in a way that's compatible with the WritingPromptDataset interface.
    """
    def __init__(self, csv_path, start_index=-1, end_index=-1, save_path=None, min_words=20):
        self.data = pd.DataFrame(columns=["id", "story_prompt", "human_story"])
        self.csv_path = csv_path
        self.min_words = min_words
        
        # Load the CSV file
        csv_data = pd.read_csv(csv_path)
        
        # Map the CSV columns to the expected format
        rows = []
        for idx, row in csv_data.iterrows():
            # Only include rows where top_story exists
            if pd.notna(row.get('top_story')):
                rows.append({
                    "id": idx,
                    "story_prompt": row.get('story_prompt', '').strip(),
                    "human_story": row.get('top_story', '').strip(),
                    # Additional fields that might be useful
                    "author": row.get('story_author', ''),
                    "upvotes": row.get('story_upvotes', 0)
                })
        
        self.data = pd.concat([self.data, pd.DataFrame(rows)], ignore_index=True)
        
        self.clean_data()
        
        if start_index != -1 and end_index != -1:
            self.data = self.data.iloc[start_index:end_index + 1].reset_index(drop=True)
        
        if save_path is not None:
            self.save_data(save_path)

    def save_data(self, save_path):
        self.data.to_csv(save_path, index=False)

    def clean_data(self):
        # Remove rows with less than min_words
        self.data['word_count'] = self.data['story_prompt'].apply(lambda x: len(str(x).split()))
        self.data = self.data[self.data['word_count'] > self.min_words]

        # Clean the content, remove [wp] tags
        self.data['story_prompt'] = self.data['story_prompt'].str.replace("[ WP ] ", "").str.replace("[ wp ] ", "")

    def get_dataframe(self):
        return pd.DataFrame(self.data)
    
    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data
    
    def get_stats_min_word_count(self):
        return self.data['word_count'].min()
    
    def get_stats_max_word_count(self):
        return self.data['word_count'].max()
    
    def get_stats_avg_word_count(self):
        return self.data['word_count'].mean()

    def get_stats(self):
        return {
            "min_word_count": self.get_stats_min_word_count(),
            "max_word_count": self.get_stats_max_word_count(),
            "avg_word_count": self.get_stats_avg_word_count()
        }
    
    def get_stat_histogram_of_word_count(self):
        word_count = self.data['word_count']
        min_words = word_count.min()
        max_words = word_count.max()
        plt.hist(word_count, bins=range(min_words, max_words + 1), edgecolor='black')
        plt.title('Histogram of Number of Words per Line')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.show()

    def __getitem__(self, idx):
        return self.data.iloc[idx]
