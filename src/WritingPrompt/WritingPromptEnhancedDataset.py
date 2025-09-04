# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Taraneh Ghandi

# If you want to use this code, please update client_id and client_secret in lines 25 and 26.

from RedditEnhancedDataset import RedditEnhancedDataset

# Initialize with the same parameters as the original class
enhanced_dataset = RedditEnhancedDataset(
    source_path=r"dataset\train.wp_source",
    target_path=r"dataset\train.wp_target",
)

# Check the current progress in the beginning
status = enhanced_dataset.get_processing_status()
print(f"Progress: {status['percentage']:.2f}% ({status['processed']}/{status['total']} items)")

# Enrich with Reddit data
enhanced_dataset.enrich_dataset_in_batches(
    client_id="",
    client_secret="",
    user_agent="python:primed_writers:v1.0 ",
    batch_size=1000,  # Adjust this based on how many items you can process in ~2 hours
    final_save_path="fully_enriched_dataset.csv"
)

# Check the current progress afterwards
status = enhanced_dataset.get_processing_status()
print(f"Progress: {status['percentage']:.2f}% ({status['processed']}/{status['total']} items)")
