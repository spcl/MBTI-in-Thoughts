# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import matplotlib.pyplot as plt
import pandas as pd

class WritingPromptDataset():
    def __init__(self, source_path, target_path, start_index=-1, end_index=-1, save_path=None, min_words=20):
        self.data = pd.DataFrame(columns=["id", "story_prompt", "human_story"])
        self.source_path = source_path
        self.target_path = target_path
        self.min_words = min_words
        
        with open(source_path, 'r', encoding='utf-8') as source_file, open(target_path, 'r', encoding='utf-8') as target_file:
            source_contents = source_file.readlines()
            target_contents = target_file.readlines()
            rows = [{"id": idx, "story_prompt": source_content.strip(), "human_story": target_content.strip()} 
                    for idx, (source_content, target_content) in enumerate(zip(source_contents, target_contents))]
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
        self.data['word_count'] = self.data['story_prompt'].apply(lambda x: len(x.split()))
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
