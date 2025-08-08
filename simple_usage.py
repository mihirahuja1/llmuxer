#!/usr/bin/env python3
"""Dead simple usage of LLMux - one line to optimize your costs."""

import llmux

# That's it! One line to find the best model for your task
result = llmux.optimize_cost(
    base_model="openai/gpt-4",
    prompt="Classify if this job posting is relevant for a Python/Rust Software Architect in SF Bay Area",
    golden_dataset="data/golden_jobs_with_baseline.jsonl",
)

print(f"Best model: {result['model']}")
print(f"Cost: ${result['cost_per_million']:.2f} per million tokens") 
if result.get('accuracy') is not None:
    print(f"Accuracy: {result['accuracy']:.1%}")

# Or optimize for speed instead of cost
fast_result = llmux.optimize_speed(
    prompt="Classify if this job posting is relevant for a Python/Rust Software Architect in SF Bay Area",
    golden_dataset="data/golden_jobs.jsonl",
)

print(f"\nFastest model: {fast_result['model']}")
print(f"Latency: {fast_result['latency']:.2f}s")
