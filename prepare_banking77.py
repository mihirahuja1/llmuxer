#!/usr/bin/env python3
"""Prepare banking77 dataset for LLMux testing.

This script downloads and converts the banking77 dataset to the expected format.
A real developer would run this once to prepare their test data.
"""

import os
import json
from datasets import load_dataset

def prepare_banking77_dataset(output_dir="data", num_samples=100):
    """Download and prepare banking77 dataset in JSONL format."""
    
    # Create data directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading banking77 dataset...")
    dataset = load_dataset("banking77", split="test")
    
    # Limit samples for faster testing
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
        print(f"Using first {num_samples} samples from test split")
    
    # Convert to JSONL format expected by LLMux
    output_path = os.path.join(output_dir, "banking77_test.jsonl")
    
    print(f"Converting to JSONL format: {output_path}")
    with open(output_path, 'w') as f:
        for item in dataset:
            # Convert to LLMux expected format
            converted_item = {
                'input': item['text'],          # Banking query text
                'label': item['label'],         # Intent category (0-76)
                'LLM_Decision': None           # Will be filled by model evaluation
            }
            f.write(json.dumps(converted_item) + '\n')
    
    print(f"✓ Banking77 dataset prepared: {output_path}")
    print(f"✓ {len(dataset)} samples ready for testing")
    
    # Show sample data
    print("\nSample data:")
    with open(output_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Show first 3 samples
                break
            sample = json.loads(line)
            print(f"  Sample {i+1}:")
            print(f"    Input: {sample['input'][:60]}...")
            print(f"    Label: {sample['label']}")
    
    return output_path

if __name__ == "__main__":
    prepare_banking77_dataset()