#!/usr/bin/env python3
"""Quick test with minimal dataset."""

import llmuxer
import os

# Set the API key if not already set
if not os.environ.get('OPENROUTER_API_KEY'):
    print("Please set OPENROUTER_API_KEY environment variable")
    print("Run: export OPENROUTER_API_KEY='your-key-here'")
    exit(1)

print("Testing LLMuxer with small dataset...")

# Use inline examples for quick testing
examples = [
    {"input": "Software Engineer at Google", "label": "Technology"},
    {"input": "Nurse at Hospital", "label": "Healthcare"},
    {"input": "Teacher at Elementary School", "label": "Education"},
    {"input": "Data Scientist at Meta", "label": "Technology"},
    {"input": "Doctor at Clinic", "label": "Healthcare"},
]

result = llmuxer.optimize_cost(
    baseline="gpt-4o-mini",
    examples=examples,
    task="classification",
    options=["Technology", "Healthcare", "Education", "Finance"],
    min_accuracy=0.6,  # Lower threshold for testing
    sample_size=1.0  # Use all 5 examples
)

if 'error' in result:
    print(f"\nError: {result['error']}")
else:
    baseline_model = "gpt-4o-mini"
    print(f"\nBest model found: {result['model']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    
    if result.get('cost_savings'):
        savings_pct = result['cost_savings'] * 100
        print(f"You save {savings_pct:.1f}% by moving from {baseline_model} to {result['model']} with {result['accuracy']:.1%} accuracy")
        print(f"New cost: ${result['cost_per_million']:.2f}/M tokens vs baseline cost")
    else:
        print(f"Cost: ${result.get('cost_per_million', 0):.2f} per million tokens")
        print(f"Switch from {baseline_model} to {result['model']} for same/better accuracy")