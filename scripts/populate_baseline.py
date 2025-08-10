#!/usr/bin/env python3
"""Populate LLM_Decision field using GPT-4 as baseline model."""

import json
import os
from llmux import get_provider

# System prompt for job classification
SYSTEM_PROMPT = """You are a job matching assistant. 
Evaluate if this job posting matches someone looking for:
- Software Architect roles (or similar senior positions)
- Python and/or Rust experience preferred
- San Francisco Bay Area location (including SF, Oakland, Palo Alto, Mountain View, etc.)

Respond with only '1' to Apply or '0' to Skip."""

def populate_llm_decisions(input_file: str, output_file: str):
    """Fill in LLM_Decision field using GPT-4."""
    
    # Initialize GPT-4 provider
    provider = get_provider("openrouter", model="openai/gpt-4o")
    
    # Load dataset
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Processing {len(data)} job postings with GPT-4...")
    
    # Process each item
    for i, item in enumerate(data):
        # Skip if already has LLM_Decision
        if item.get('LLM_Decision') is not None:
            print(f"  [{i+1}/{len(data)}] Skipping (already has decision)")
            continue
        
        # Get GPT-4 decision
        prompt = f"{SYSTEM_PROMPT}\n\n{item['input']}\n\nRespond with only '1' to Apply or '0' to Skip."
        
        try:
            response = provider.complete(prompt).strip()
            decision = 1 if '1' in response else 0
            item['LLM_Decision'] = decision
            
            first_line = item['input'].split('\n')[0][:50]
            print(f"  [{i+1}/{len(data)}] Decision: {decision} - {first_line}...")
            
        except Exception as e:
            print(f"  [{i+1}/{len(data)}] Error: {e}")
            item['LLM_Decision'] = None
    
    # Save updated dataset
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    # Summary
    decisions = [item['LLM_Decision'] for item in data if item['LLM_Decision'] is not None]
    print(f"\n✅ Populated {len(decisions)} decisions")
    print(f"   Apply (1): {decisions.count(1)}")
    print(f"   Skip (0): {decisions.count(0)}")
    print(f"   Saved to: {output_file}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        exit(1)
    
    # Process the golden dataset
    populate_llm_decisions(
        input_file="data/golden_jobs.jsonl",
        output_file="data/golden_jobs_with_baseline.jsonl"
    )
    
    print("\n💡 Next steps:")
    print("1. Review the decisions in data/golden_jobs_with_baseline.jsonl")
    print("2. Add your Human_Input labels where GPT-4 was wrong")
    print("3. Use this dataset to find cheaper models that match GPT-4's accuracy")