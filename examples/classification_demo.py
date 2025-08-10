#!/usr/bin/env python3
"""Banking77 dataset test."""

import llmux
import os

dataset_path = "data/banking77_test.jsonl"
if not os.path.exists(dataset_path):
    print("Dataset not found. Run: python prepare_banking77.py")
    exit(1)

result = llmux.optimize_cost(
    baseline="gpt-4",
    dataset=dataset_path,
    prompt="Classify the banking customer query into one of 77 intent categories (0-76).",
    task="classification",
    min_accuracy=0.7,
    sample_size=0.2  # Use only 20% of the dataset for faster testing
)

print(result)
