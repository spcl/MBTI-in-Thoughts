# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

import os
import json
import praw
import time

from difflib import SequenceMatcher
from tqdm import tqdm
from WritingPromptDataset import WritingPromptDataset

class RedditEnhancedDataset(WritingPromptDataset):
    """
    Enhanced version of WritingPromptDataset that includes Reddit API integration
    to fetch additional data like upvotes and comment counts.
    """
    
    def __init__(self, source_path, target_path, start_index=-1, end_index=-1, save_path=None, min_words=20):
        """Initialize with the same parameters as the parent class"""
        super().__init__(source_path, target_path, start_index, end_index, save_path, min_words)
        self.reddit = None
        self.writingprompts_subreddit = None
        self._checkpoint_file = "reddit_enrichment_checkpoint.json"
        
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
    
    def match_story_with_comment(self, story_text, comments):
        """
        Try to match a story in the dataset with a comment from Reddit.
        
        Args:
            story_text (str): The story text from the dataset
            comments (list): List of Reddit comments to check against
            
        Returns:
            praw.models.Comment or None: The matched comment or None if not found
        """
        # Simple matching: check first few words
        story_start = ' '.join(story_text.split()[:20]).lower()
        
        for comment in comments:
            if hasattr(comment, 'body') and len(comment.body) > 0:
                comment_start = ' '.join(comment.body.split()[:20]).lower().replace("'", "'").replace('"', '').replace('\\', '')

                # Calculate similarity ratio instead of simple matching
                similarity = SequenceMatcher(None, story_start, comment_start).ratio()

                # Use a threshold for matching (65% similarity)
                if similarity >= 0.65:
                    return comment
        
        return None
    
    def _save_checkpoint(self, processed_indices, checkpoint_file=None):
        """Save the current processing state to resume later"""
        if checkpoint_file is None:
            checkpoint_file = self._checkpoint_file
            
        checkpoint_data = {
            "processed_indices": processed_indices,
            "total_rows": len(self.data)
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved: {len(processed_indices)}/{checkpoint_data['total_rows']} rows processed")
    
    def _load_checkpoint(self, checkpoint_file=None):
        """Load the previous processing state"""
        if checkpoint_file is None:
            checkpoint_file = self._checkpoint_file
            
        if not os.path.exists(checkpoint_file):
            return set()
            
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        processed_indices = set(checkpoint_data["processed_indices"])
        print(f"Checkpoint loaded: {len(processed_indices)}/{checkpoint_data['total_rows']} rows already processed")
        return processed_indices
    
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
    
    def enrich_dataset_in_batches(self, client_id, client_secret, user_agent, 
                                 batch_size=1000, checkpoint_file=None, 
                                 save_interim=True, final_save_path=None):
        """
        Process the dataset in batches with checkpointing to handle VPN time constraints.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent string for API requests
            batch_size (int): Number of items to process before checkpointing
            checkpoint_file (str): Path to save checkpoint data
            save_interim (bool): Whether to save the dataset after each batch
            final_save_path (str): Path to save the final enriched dataset
            
        Returns:
            RedditEnhancedDataset: The current instance with enriched data
        """
        # Initialize Reddit API if not already initialized
        if not self.reddit:
            self.initialize_reddit_api(client_id, client_secret, user_agent)
        
        # Set up checkpoint file
        if checkpoint_file:
            self._checkpoint_file = checkpoint_file
        
        # Add new columns if they don't exist
        if 'post_id' not in self.data.columns:
            self.data['post_id'] = None
            self.data['prompt_upvotes'] = None
            self.data['prompt_comments_count'] = None
            self.data['story_upvotes'] = None
            self.data['story_author'] = None
        
        # Load checkpoint to resume processing
        processed_indices = self._load_checkpoint()
        
        # Get indices of rows that still need processing
        remaining_indices = [i for i in range(len(self.data)) if i not in processed_indices]
        
        if not remaining_indices:
            print("All data already processed!")
            return self
        
        # Group rows by unique prompts to avoid duplicate API calls
        prompt_to_indices = {}
        for idx in remaining_indices:
            prompt = self.data.iloc[idx]['story_prompt']
            if prompt not in prompt_to_indices:
                prompt_to_indices[prompt] = []
            prompt_to_indices[prompt].append(idx)
        
        unique_prompts = list(prompt_to_indices.keys())
        print(f"Found {len(unique_prompts)} unique prompts out of {len(remaining_indices)} remaining rows")
        
        # Process in batches of unique prompts
        total_batches = (len(unique_prompts) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, len(unique_prompts))
            batch_prompts = unique_prompts[batch_start:batch_end]
            
            # Count total rows in this batch for time estimation
            batch_rows_count = sum(len(prompt_to_indices[prompt]) for prompt in batch_prompts)
            
            # Estimate time for this batch
            estimated_time = self.estimate_processing_time(len(batch_prompts))
            print(f"\nProcessing batch {batch_num+1}/{total_batches} ({len(batch_prompts)} unique prompts, {batch_rows_count} total rows)")
            print(f"Estimated time: {estimated_time}")
            
            # Process each unique prompt in the current batch
            newly_processed = []
            for prompt_idx, prompt_text in enumerate(tqdm(batch_prompts, desc=f"Batch {batch_num+1}")):
                # Get all indices in the dataset with this prompt
                indices = prompt_to_indices[prompt_text]
                
                # Find the prompt post on Reddit (only once per unique prompt)
                post = self.find_prompt_post(prompt_text)
                
                if post:
                    # Store post info for all rows with this prompt
                    post_id = post.id
                    prompt_upvotes = post.score
                    prompt_comments_count = post.num_comments
                    
                    # Load all comments once
                    post.comments.replace_more(limit=None)
                    all_comments = list(post.comments.list())
                    
                    # Process each story for this prompt
                    for idx in indices:
                        row = self.data.iloc[idx]
                        human_story = row['human_story']
                        
                        # Store prompt info
                        self.data.at[idx, 'post_id'] = post_id
                        self.data.at[idx, 'prompt_upvotes'] = prompt_upvotes
                        self.data.at[idx, 'prompt_comments_count'] = prompt_comments_count
                        
                        # Try to match the story with a Reddit comment
                        matched_comment = self.match_story_with_comment(human_story, all_comments)
                        
                        if matched_comment:
                            self.data.at[idx, 'story_upvotes'] = matched_comment.score
                            self.data.at[idx, 'story_author'] = (
                                matched_comment.author.name if matched_comment.author else "[deleted]"
                            )
                        
                        newly_processed.append(idx)
                else:
                    # If prompt not found, mark all associated rows as processed
                    newly_processed.extend(indices)
                
                # Sleep to avoid hitting Reddit API rate limits
                time.sleep(1)
                
                # Save an intermediate checkpoint every 10 prompts
                if (prompt_idx + 1) % 10 == 0:
                    processed_indices.update(newly_processed)
                    self._save_checkpoint(list(processed_indices))
                    newly_processed = []
            
            # Update and save checkpoint after each batch
            processed_indices.update(newly_processed)
            self._save_checkpoint(list(processed_indices))
            
            # Save interim dataset if requested
            if save_interim:
                interim_save_path = f"enriched_dataset_batch_{batch_num+1}.csv"
                self.save_data(interim_save_path)
                print(f"Interim dataset saved to {interim_save_path}")
            
            # If there are more batches, inform the user
            if batch_num < total_batches - 1:
                print("\n" + "="*50)
                print(f"Batch {batch_num+1} completed! You can now turn on your VPN if needed.")
                print(f"To continue with the next batch when ready, run this function again.")
                print("="*50 + "\n")
                
                # Early exit after batch is done to allow VPN reconnection
                return self
        
        # Save final enriched dataset
        if final_save_path:
            self.save_data(final_save_path)
            print(f"Final enriched dataset saved to {final_save_path}")
            
        return self
    
    def get_popular_stories(self, min_upvotes=10):
        """
        Get stories that have received at least a specified number of upvotes.
        
        Args:
            min_upvotes (int): Minimum number of upvotes required
            
        Returns:
            pandas.DataFrame: Filtered dataset with popular stories
        """
        if 'story_upvotes' not in self.data.columns:
            raise ValueError("Dataset not yet enriched with Reddit data. Call enrich_dataset first.")
        
        return self.data[self.data['story_upvotes'] >= min_upvotes]
    
    def get_engagement_stats(self):
        """
        Get statistics about engagement with the prompts and stories.
        
        Returns:
            dict: Dictionary containing engagement statistics
        """
        if 'prompt_upvotes' not in self.data.columns:
            raise ValueError("Dataset not yet enriched with Reddit data. Call enrich_dataset first.")
        
        return {
            "avg_prompt_upvotes": self.data['prompt_upvotes'].mean(),
            "max_prompt_upvotes": self.data['prompt_upvotes'].max(),
            "avg_comments": self.data['prompt_comments_count'].mean(),
            "avg_story_upvotes": self.data['story_upvotes'].mean(),
            "successful_matches": (self.data['story_upvotes'].notnull().sum() / len(self.data)) * 100
        }
        
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
