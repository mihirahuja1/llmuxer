#!/usr/bin/env python3
"""Generate model table from curated MODEL_UNIVERSE for README."""

import sys
import os

# Add parent directory to path to import llmuxer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llmuxer.api import fetch_openrouter_models


def get_model_universe():
    """Get the curated model universe from API."""
    MODEL_UNIVERSE = {
        # OpenAI - 2 models
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo",
        # Anthropic - 2 models  
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-sonnet",
        # Google - 3 models
        "google/gemini-1.5-flash-8b", 
        "google/gemini-1.5-flash",
        "google/gemini-1.5-pro",
        # Qwen - 3 models
        "alibaba/qwen-2.5-7b-instruct",
        "alibaba/qwen-2.5-14b-instruct", 
        "alibaba/qwen-2.5-72b-instruct",
        # DeepSeek - 3 models
        "deepseek/deepseek-chat",
        "deepseek/deepseek-coder", 
        "deepseek/deepseek-v2.5",
        # Mistral - 3 models
        "mistralai/mistral-7b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "mistralai/mistral-large",
        # Meta - 2 models
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct"
    }
    return MODEL_UNIVERSE


def generate_model_table():
    """Generate markdown table of models and pricing."""
    print("Fetching live model pricing from OpenRouter...")
    
    try:
        model_costs = fetch_openrouter_models()
        universe = get_model_universe()
        
        # Group by provider
        providers = {
            "OpenAI": [],
            "Anthropic": [],
            "Google": [],
            "Qwen": [],
            "DeepSeek": [],
            "Mistral": [],
            "Meta": []
        }
        
        for model_id in universe:
            if model_id in model_costs:
                cost_data = model_costs[model_id]
                total_cost = cost_data["input"] + cost_data["output"]
                
                model_name = model_id.split("/")[-1]
                
                if model_id.startswith("openai/"):
                    providers["OpenAI"].append((model_name, total_cost))
                elif model_id.startswith("anthropic/"):
                    providers["Anthropic"].append((model_name, total_cost))
                elif model_id.startswith("google/"):
                    providers["Google"].append((model_name, total_cost))
                elif model_id.startswith("alibaba/qwen") or "qwen" in model_id.lower():
                    providers["Qwen"].append((model_name, total_cost))
                elif model_id.startswith("deepseek/"):
                    providers["DeepSeek"].append((model_name, total_cost))
                elif model_id.startswith("mistralai/"):
                    providers["Mistral"].append((model_name, total_cost))
                elif model_id.startswith("meta-llama/"):
                    providers["Meta"].append((model_name, total_cost))
        
        # Generate table
        print("\n## Tested Models\n")
        print("| Provider | Models | Price Range ($/M tokens) |")
        print("|----------|--------|---------------------------|")
        
        total_models = 0
        for provider, models in providers.items():
            if models:
                models.sort(key=lambda x: x[1])  # Sort by cost
                model_names = [m[0] for m in models]
                costs = [m[1] for m in models]
                
                if len(costs) > 1:
                    price_range = f"${min(costs):.2f} - ${max(costs):.2f}"
                else:
                    price_range = f"${costs[0]:.2f}"
                
                models_str = ", ".join(model_names)
                if len(models_str) > 50:
                    models_str = f"{len(models)} models"
                
                print(f"| {provider} | {models_str} | {price_range} |")
                total_models += len(models)
        
        print(f"\n**Total: {total_models} models across {len([p for p, m in providers.items() if m])} providers**")
        
        # Generate benchmark command
        print(f"\n### Reproduce Our Benchmarks")
        print(f"```bash")
        print(f"# Test all {total_models} models on Banking77 dataset")
        print(f"python scripts/prepare_banking77.py")
        print(f"python examples/banking77_test.py")
        print(f"```")
        
    except Exception as e:
        print(f"Error fetching model data: {e}")
        print("Using fallback static table...")
        
        # Fallback static table
        print("| Provider | Models | Price Range ($/M tokens) |")
        print("|----------|--------|---------------------------|")
        print("| OpenAI | gpt-4o-mini, gpt-3.5-turbo | $0.75 - $2.00 |")
        print("| Anthropic | claude-3-haiku, claude-3-sonnet | $0.25 - $3.00 |")
        print("| Google | gemini-1.5 (3 variants) | $0.04 - $1.25 |")
        print("| Qwen | 2.5 (7b, 14b, 72b) | $0.10 - $0.70 |")
        print("| DeepSeek | chat, coder, v2.5 | $0.14 - $0.28 |")
        print("| Mistral | 7b, mixtral-8x7b, large | $0.06 - $2.00 |")
        print("| Meta | llama-3.1 (8b, 70b) | $0.06 - $0.35 |")
        print("\n**Total: 18 models across 7 providers**")


if __name__ == "__main__":
    generate_model_table()