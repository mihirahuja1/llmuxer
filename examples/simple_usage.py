#!/usr/bin/env python3
"""Simple LLMuxer usage example."""

import llmuxer

print("Finding cheaper model than gpt-4o-mini...")

# Super simple - just baseline and dataset, everything else is auto-detected
result = llmuxer.optimize_cost(
    baseline="gpt-4o-mini", 
    dataset="../data/golden_jobs_with_baseline.jsonl",
    min_accuracy=0.7
)

if 'error' in result:
    print(f"Error: {result['error']}")
else:
    print(f"\nBest model: {result['model']}")
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

# # === More Examples (Commented Out) ===

# # Classification with explicit success criteria
# sentiment_examples = [
#     {"input": "I love this product!", "label": "positive"},
#     {"input": "It's okay, nothing special", "label": "neutral"},
#     {"input": "Terrible quality, would not recommend", "label": "negative"},
# ]
# result = llmuxer.optimize_cost(
#     baseline="gpt-4",
#     examples=sentiment_examples,
#     task="classification",
#     options=["positive", "negative", "neutral"]
# )

# # Extraction task
# name_examples = [
#     {"input": "Hi, I'm John Smith from Google", "label": "John Smith"},
#     {"input": "Sarah Johnson will be joining us", "label": "Sarah Johnson"},
#     {"input": "The meeting is at 3pm", "label": None},
# ]
# result = llmuxer.optimize_cost(
#     baseline="gpt-4",
#     examples=name_examples,
#     task="extraction",
#     extract="person_name"
# )

# # CSV support
# result = llmuxer.optimize_cost(
#     baseline="gpt-4",
#     dataset="reviews.csv",
#     task="classification",
#     options=["positive", "negative", "neutral"]
# )

# # HuggingFace dataset
# result = llmuxer.optimize_cost(
#     baseline="gpt-4",
#     dataset="imdb:test[:100]",
#     task="binary",
#     options=["positive", "negative"]
# )