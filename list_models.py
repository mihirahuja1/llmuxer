#!/usr/bin/env python3
"""List all available models from OpenRouter."""

import requests
import json

def list_available_models():
    """Fetch and display all available models with their pricing."""
    
    print("Fetching available models from OpenRouter...")
    
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        models_data = response.json()
        
        # Organize models by provider
        models_by_provider = {}
        
        for model in models_data.get('data', []):
            model_id = model['id']
            
            # Skip special/auto models
            if model_id in ['openrouter/auto', 'openrouter/best']:
                continue
            
            # Get pricing info
            pricing = model.get('pricing', {})
            if not pricing or 'prompt' not in pricing:
                continue
                
            prompt_cost = float(pricing.get('prompt', 0)) * 1_000_000
            completion_cost = float(pricing.get('completion', 0)) * 1_000_000
            
            # Skip free or invalid pricing
            if prompt_cost < 0 or completion_cost < 0:
                continue
                
            # Extract provider from model ID
            provider = model_id.split('/')[0] if '/' in model_id else 'unknown'
            
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            
            models_by_provider[provider].append({
                'id': model_id,
                'name': model.get('name', model_id),
                'input_cost': prompt_cost,
                'output_cost': completion_cost,
                'total_cost': prompt_cost + completion_cost,
                'context': model.get('context_length', 0)
            })
        
        # Sort providers and models
        for provider in sorted(models_by_provider.keys()):
            models = models_by_provider[provider]
            # Sort by total cost
            models.sort(key=lambda x: x['total_cost'])
            
            print(f"\n{'='*80}")
            print(f"PROVIDER: {provider.upper()}")
            print(f"{'='*80}")
            print(f"{'Model':<40} {'Input $/M':<12} {'Output $/M':<12} {'Context':<10}")
            print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*10}")
            
            for model in models[:10]:  # Show top 10 per provider
                model_name = model['id'].split('/')[-1] if '/' in model['id'] else model['id']
                print(f"{model_name:<40} ${model['input_cost']:<11.3f} ${model['output_cost']:<11.3f} {model['context']:<10,}")
        
        # Show summary
        total_models = sum(len(models) for models in models_by_provider.values())
        print(f"\n{'='*80}")
        print(f"SUMMARY: {total_models} models available from {len(models_by_provider)} providers")
        print(f"{'='*80}")
        
        # Show cheapest models overall
        all_models = []
        for models in models_by_provider.values():
            all_models.extend(models)
        all_models.sort(key=lambda x: x['total_cost'])
        
        print("\nTOP 10 CHEAPEST MODELS:")
        print(f"{'Model':<50} {'Total $/M':<15}")
        print(f"{'-'*50} {'-'*15}")
        for model in all_models[:10]:
            print(f"{model['id']:<50} ${model['total_cost']:<14.3f}")
            
    except Exception as e:
        print(f"Error fetching models: {e}")
        print("\nFallback models typically available:")
        print("- openai/gpt-4")
        print("- openai/gpt-4-turbo") 
        print("- openai/gpt-3.5-turbo")
        print("- anthropic/claude-3-opus")
        print("- anthropic/claude-3-sonnet")
        print("- anthropic/claude-3-haiku")
        print("- google/gemini-pro")
        print("- google/gemini-flash")
        print("- meta-llama/llama-3.1-405b")
        print("- meta-llama/llama-3.1-70b")
        print("- meta-llama/llama-3.1-8b")
        print("- mistralai/mistral-7b-instruct")

if __name__ == "__main__":
    list_available_models()