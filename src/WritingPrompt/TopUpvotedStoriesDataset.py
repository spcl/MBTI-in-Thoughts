# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import datetime
import json
import os
import praw
import time

from difflib import SequenceMatcher
import pandas as pd
from tqdm import tqdm
from WritingPromptDataset import WritingPromptDataset

class TopUpvotedStoriesDataset(WritingPromptDataset):
    """
    A dataset class that finds the most upvoted story (pre-2020) for each writing prompt.
    Extends WritingPromptDataset and fetches data from Reddit.
    Only keeps one instance of each unique story prompt.
    """
    def __init__(self, source_path, target_path=None, start_index=-1, end_index=-1, save_path=None, min_words=20):
        """
        Initialize with source path (and optionally target path) to get writing prompts.
        If target_path is None, it will only load prompts without associated stories.
        """
        if target_path:
            super().__init__(source_path, target_path, start_index, end_index, save_path, min_words)
            # Deduplicate - keep only first occurrence of each prompt
            self.data = self.data.drop_duplicates(subset=['story_prompt'], keep='first').reset_index(drop=True)
        else:
            # If no target_path, just load prompts
            self.data = pd.DataFrame(columns=["id", "story_prompt"])
            self.source_path = source_path
            self.min_words = min_words
            
            with open(source_path, 'r', encoding='utf-8') as source_file:
                source_contents = source_file.readlines()
                rows = [{"id": idx, "story_prompt": source_content.strip()} 
                        for idx, source_content in enumerate(source_contents)]
                self.data = pd.concat([self.data, pd.DataFrame(rows)], ignore_index=True)
            
            self.clean_data()
            
            # Deduplicate - keep only first occurrence of each prompt
            self.data = self.data.drop_duplicates(subset=['story_prompt'], keep='first').reset_index(drop=True)
            
            if start_index != -1 and end_index != -1:
                self.data = self.data.iloc[start_index:end_index + 1].reset_index(drop=True)
            
            if save_path is not None:
                self.save_data(save_path)
                
        # Reddit API related attributes
        self.reddit = None
        self.writingprompts_subreddit = None
        self._checkpoint_file = "top_stories_checkpoint.json"
        
        # Add columns for the top upvoted stories
        if 'top_story' not in self.data.columns:
            self.data['top_story'] = None
            self.data['post_id'] = None
            self.data['prompt_upvotes'] = None
            self.data['prompt_comments_count'] = None
            self.data['story_upvotes'] = None
            self.data['story_author'] = None
            self.data['story_timestamp'] = None
    
    def initialize_reddit_api(self, client_id, client_secret, user_agent):
        """
        Initialize the Reddit API connection using PRAW.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent string for API requests
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.writingprompts_subreddit = self.reddit.subreddit('WritingPrompts')
        print("Reddit API initialized successfully")
        return self
    
    def find_prompt_post(self, prompt_text, max_search_results=100):
        """
        Search for a writing prompt on Reddit that matches the given text.
        
        Args:
            prompt_text (str): The prompt text to search for
            max_search_results (int): Maximum number of search results to check
            
        Returns:
            praw.models.Submission or None: The matched Reddit post or None if not found
        """
        if not self.reddit:
            raise RuntimeError("Reddit API not initialized. Call initialize_reddit_api first.")
        
        # Create search terms from prompt (use first few words)
        search_terms = ' '.join(prompt_text.split()[:8])
        
        try:
            # Search WritingPrompts subreddit
            search_results = self.writingprompts_subreddit.search(search_terms, limit=max_search_results)
            
            for post in search_results:
                # Clean post title for comparison
                post_title = post.title.replace("[WP]", "").replace("[wp]", "").strip()
                # Clean both texts for comparison
                clean_prompt = prompt_text.lower().replace("'", "'").replace('"', '').replace('\\', '')
                clean_title = post_title.lower().replace("'", "'").replace('"', '').replace('\\', '')

                # Calculate similarity ratio
                similarity = SequenceMatcher(None, clean_prompt, clean_title).ratio()
                
                # Use a threshold for matching (65% similarity)
                if similarity >= 0.65:
                    return post
                # For debugging purposes
                elif similarity >= 0.6:
                    print(f"Potential match: {similarity:.2f}\nPrompt: {clean_prompt}\nTitle: {clean_title}")
            
            return None
        except Exception as e:
            print(f"Error searching for prompt: {e}")
            return None
    
    def get_top_pre_2020_story(self, post):
        """
        Find the most upvoted story comment from before 2020 for a Reddit post.
        
        Args:
            post (praw.models.Submission): The Reddit post to analyze
            
        Returns:
            tuple: (comment, comment_created_time) or (None, None) if no valid comment found
        """
        if not post:
            return None, None
        
        # Load all comments
        post.comments.replace_more(limit=None)
        all_comments = list(post.comments.list())
        
        # Filter for pre-2020 comments
        pre_2020_limit = datetime.datetime(2020, 1, 1, 0, 0, 0).timestamp()
        
        # Sort by score (upvotes) and filter by date
        valid_comments = []
        for comment in all_comments:
            if hasattr(comment, 'created_utc') and comment.created_utc < pre_2020_limit:
                if hasattr(comment, 'body') and len(comment.body) > 20:  # Ensure it's a substantial comment
                    valid_comments.append(comment)
        
        # Sort by upvotes (highest first)
        valid_comments.sort(key=lambda x: x.score, reverse=True)
        
        # Return the top comment and its creation time
        if valid_comments:
            top_comment = valid_comments[0]
            created_time = datetime.datetime.fromtimestamp(top_comment.created_utc)
            return top_comment, created_time
        
        return None, None
    
    def _save_checkpoint(self, processed_indices, checkpoint_file=None, current_batch=0):
        """Save the current processing state to resume later"""
        if checkpoint_file is None:
            checkpoint_file = self._checkpoint_file
            
        checkpoint_data = {
            "processed_indices": processed_indices,
            "total_rows": len(self.data),
            "last_batch": current_batch  # Save the current batch number
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved: {len(processed_indices)}/{checkpoint_data['total_rows']} rows processed (Batch {current_batch})")
    
    def _load_checkpoint(self, checkpoint_file=None):
        """Load the previous processing state"""
        if checkpoint_file is None:
            checkpoint_file = self._checkpoint_file
            
        if not os.path.exists(checkpoint_file):
            return set(), 0  # Return empty set and batch 0
            
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        processed_indices = set(checkpoint_data["processed_indices"])
        # Get the last batch number, default to 0 if not found (for backward compatibility)
        last_batch = checkpoint_data.get("last_batch", 0)
        
        print(f"Checkpoint loaded: {len(processed_indices)}/{checkpoint_data['total_rows']} rows processed (Last completed batch: {last_batch})")
        return processed_indices, last_batch
    
    def estimate_processing_time(self, batch_size):
        """Estimate the time needed to process a batch"""
        # Assume 2 seconds per item (API call + processing)
        seconds_per_item = 2
        estimated_seconds = batch_size * seconds_per_item
        
        # Convert to hours, minutes, seconds
        hours = estimated_seconds // 3600
        minutes = (estimated_seconds % 3600) // 60
        seconds = estimated_seconds % 60
        
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    def save_batch_data(self, save_path, batch_indices):
        """
        Save only the data from the current batch to a CSV file.
        
        Args:
            save_path (str): Path to save the batch data
            batch_indices (list): List of indices processed in this batch
        """
        if not batch_indices:
            print("No data to save in this batch")
            return
            
        # Get only the rows processed in this batch
        batch_data = self.data.iloc[batch_indices].copy()
        
        # Only include rows where we actually found data
        has_data = batch_data['post_id'].notna() | batch_data['top_story'].notna()
        batch_data = batch_data[has_data]
        
        if len(batch_data) == 0:
            print("No data found for any prompts in this batch")
            return
            
        # Save to CSV with all columns
        batch_data.to_csv(save_path, index=False)
        print(f"Saved {len(batch_data)} entries with data to {save_path}")
        
        # Print sample to verify data is being saved correctly
        if not batch_data.empty:
            sample = batch_data.sample(min(1, len(batch_data)))
            print("\nSample of saved data:")
            for _, row in sample.iterrows():
                print(f"Prompt: {row['story_prompt'][:50]}...")
                print(f"Has story: {'Yes' if pd.notna(row['top_story']) else 'No'}")
                print(f"Story upvotes: {row['story_upvotes']}")
                print(f"Post ID: {row['post_id']}")

    def fetch_top_stories_in_batches(self, client_id, client_secret, user_agent, 
                                    batch_size=1000, checkpoint_file=None, 
                                    save_interim=True, final_save_path=None):
        """
        Process the dataset in batches, finding top pre-2020 stories for each prompt.
        Dataset is already deduplicated, so each prompt is processed only once.
        Maintains batch numbering across multiple runs and saves only new data.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent string for API requests
            batch_size (int): Number of items to process before checkpointing
            checkpoint_file (str): Path to save checkpoint data
            save_interim (bool): Whether to save the dataset after each batch
            final_save_path (str): Path to save the final enriched dataset
            
        Returns:
            TopUpvotedStoriesDataset: The current instance with fetched top stories
        """
        # Initialize Reddit API if not already initialized
        if not self.reddit:
            self.initialize_reddit_api(client_id, client_secret, user_agent)
        
        # Set up checkpoint file
        if checkpoint_file:
            self._checkpoint_file = checkpoint_file
        
        # Load checkpoint to resume processing
        processed_indices, last_batch = self._load_checkpoint()
        
        # Get indices of rows that still need processing
        remaining_indices = [i for i in range(len(self.data)) if i not in processed_indices]
        
        if not remaining_indices:
            print("All data already processed!")
            return self
            
        # Process in batches
        total_batches = (len(remaining_indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            # Calculate the actual batch number (continuing from last run)
            batch_num = last_batch + batch_idx + 1
            
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(remaining_indices))
            batch_indices = remaining_indices[batch_start:batch_end]
            
            # Keep track of indices processed in this specific batch for saving
            current_batch_processed = []
            
            # Estimate time for this batch
            estimated_time = self.estimate_processing_time(len(batch_indices))
            print(f"\nProcessing batch {batch_num} (Batch {batch_idx+1}/{total_batches} of current run) with {len(batch_indices)} prompts")
            print(f"Estimated time: {estimated_time}")
            
            # Process each prompt in the current batch
            newly_processed = []
            for i, idx in enumerate(tqdm(batch_indices, desc=f"Batch {batch_num}")):
                prompt_text = self.data.iloc[idx]['story_prompt']
                
                # Find the prompt post on Reddit
                post = self.find_prompt_post(prompt_text)
                
                if post:
                    # Store post info
                    self.data.at[idx, 'post_id'] = post.id
                    self.data.at[idx, 'prompt_upvotes'] = post.score
                    self.data.at[idx, 'prompt_comments_count'] = post.num_comments
                    
                    # Get the most upvoted pre-2020 story
                    top_comment, comment_time = self.get_top_pre_2020_story(post)
                    
                    # Store top story info if found
                    if top_comment:
                        self.data.at[idx, 'top_story'] = top_comment.body
                        self.data.at[idx, 'story_upvotes'] = top_comment.score
                        self.data.at[idx, 'story_author'] = (
                            top_comment.author.name if top_comment.author else "[deleted]"
                        )
                        self.data.at[idx, 'story_timestamp'] = comment_time.strftime('%Y-%m-%d %H:%M:%S')
                
                newly_processed.append(idx)
                current_batch_processed.append(idx)
                
                # Sleep to avoid hitting Reddit API rate limits
                time.sleep(1)
                
                # Save an intermediate checkpoint every 10 prompts
                if (i + 1) % 10 == 0:
                    processed_indices.update(newly_processed)
                    self._save_checkpoint(list(processed_indices), current_batch=batch_num)
                    newly_processed = []
            
            # Update and save checkpoint after each batch
            processed_indices.update(newly_processed)
            self._save_checkpoint(list(processed_indices), current_batch=batch_num)
            
            # Save interim dataset if requested - only save the current batch data
            if save_interim:
                # Include batch number in the filename to avoid overwriting
                interim_save_path = f"top_stories_batch_{batch_num}.csv"
                self.save_batch_data(interim_save_path, current_batch_processed)
            
            # If there are more batches, inform the user
            if batch_idx < total_batches - 1:
                print("\n" + "="*50)
                print(f"Batch {batch_num} completed! You can now turn on your VPN if needed.")
                print(f"To continue with the next batch when ready, run this function again.")
                print("="*50 + "\n")
                
                # Early exit after batch is done to allow VPN reconnection
                return self
        
        # Save final enriched dataset - this will contain ALL processed data
        if final_save_path:
            # For the final dataset, include only rows that have data
            final_data = self.data.copy()
            has_data = final_data['post_id'].notna() | final_data['top_story'].notna()
            final_data = final_data[has_data]
            
            final_data.to_csv(final_save_path, index=False)
            print(f"Final enriched dataset saved to {final_save_path} with {len(final_data)} entries")
            
        return self
    
    def get_popular_stories(self, min_upvotes=10):
        """
        Get top stories that have received at least a specified number of upvotes.
        
        Args:
            min_upvotes (int): Minimum number of upvotes required
            
        Returns:
            pandas.DataFrame: Filtered dataset with popular stories
        """
        if 'story_upvotes' not in self.data.columns:
            raise ValueError("Dataset not yet processed with top stories. Call fetch_top_stories_in_batches first.")
        
        return self.data[self.data['story_upvotes'] >= min_upvotes]
    
    def get_processing_status(self):
        """Get the current processing status"""
        if os.path.exists(self._checkpoint_file):
            with open(self._checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                
            processed = len(checkpoint_data["processed_indices"])
            total = checkpoint_data["total_rows"]
            percentage = (processed / total) * 100 if total > 0 else 0
            
            return {
                "processed": processed,
                "total": total,
                "percentage": percentage,
                "remaining": total - processed
            }
        else:
            return {
                "processed": 0,
                "total": len(self.data),
                "percentage": 0,
                "remaining": len(self.data)
            }
    
    def get_engagement_stats(self):
        """
        Get statistics about engagement with the prompts and top stories.
        
        Returns:
            dict: Dictionary containing engagement statistics
        """
        if 'prompt_upvotes' not in self.data.columns:
            raise ValueError("Dataset not yet processed with top stories. Call fetch_top_stories_in_batches first.")
        
        # Filter out rows where we actually found a story
        stories_found = self.data[self.data['top_story'].notnull()]
        
        return {
            "avg_prompt_upvotes": self.data['prompt_upvotes'].mean(),
            "max_prompt_upvotes": self.data['prompt_upvotes'].max(),
            "avg_comments": self.data['prompt_comments_count'].mean(),
            "avg_story_upvotes": stories_found['story_upvotes'].mean() if not stories_found.empty else 0,
            "successful_matches": (stories_found.shape[0] / len(self.data)) * 100,
            "pre_2020_stories_found": stories_found.shape[0]
        }
