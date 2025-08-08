#!/usr/bin/env python3
"""Banking77 dataset test for LLMux - Intent classification benchmark."""

import llmux
import os

print("Banking77 Intent Classification Benchmark")
print("=========================================")

# Check if dataset exists, if not, prepare it
dataset_path = "data/banking77_test.jsonl"
if not os.path.exists(dataset_path):
    print("Dataset not found. Please run: python prepare_banking77.py")
    print("This will download and prepare the banking77 dataset locally.")
    exit(1)

print("Finding cheaper models than gpt-4 for banking intent classification...")

# Single optimize_cost call to find the best alternative to gpt-4
result = llmux.optimize_cost(
    baseline="gpt-4", 
    dataset=dataset_path,  # Local banking77 dataset file
    task="classification",  # This is a classification task
    min_accuracy=0.7  # Allow slightly lower accuracy for cost savings
)

if 'error' in result:
    print(f"Error: {result['error']}")
    print("Note: You may need to install datasets: pip install datasets")
else:
    print(f"\nBest Model Found:")
    print(f"Model: {result['model']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    
    if result.get('cost_savings'):
        savings_pct = result['cost_savings'] * 100
        cost_per_m = result['cost_per_million']
        print(f"Cost savings: {savings_pct:.1f}% (${cost_per_m:.2f}/M tokens)")
        print(f"Why picked: Best cost-efficiency ratio (accuracy/cost)")
    else:
        cost_per_m = result.get('cost_per_million', 0)
        print(f"Cost: ${cost_per_m:.2f} per million tokens")
    
    if result.get('below_threshold'):
        print(f"Note: Accuracy {result['accuracy']:.1%} is below target {result['threshold']:.0%}")

print("\n" + "="*50)
print("Banking77 Dataset Info:")
print("- 77 different banking intent categories")
print("- Customer service queries and banking operations")
print("- High-quality labeled data for intent classification")
print("- Test subset: 100 samples for quick evaluation")