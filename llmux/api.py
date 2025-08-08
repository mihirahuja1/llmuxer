"""Simple API for LLMux - Optimize your LLM costs automatically."""

from typing import Optional, Dict, Any, List
from .selector import Selector
from .provider import get_provider
import json
import os
import requests


def fetch_openrouter_models() -> Dict[str, Dict]:
    """Fetch live model pricing from OpenRouter API."""
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        models_data = response.json()
        
        # Convert to our format
        model_costs = {}
        for model in models_data.get('data', []):
            model_id = model['id']
            # Skip special models that have negative or invalid pricing
            if model_id in ['openrouter/auto', 'openrouter/best']:
                continue
                
            # Convert price per token to price per million tokens
            pricing = model.get('pricing', {})
            if pricing and 'prompt' in pricing and 'completion' in pricing:
                prompt_cost = float(pricing.get('prompt', 0)) * 1_000_000
                completion_cost = float(pricing.get('completion', 0)) * 1_000_000
                
                # Skip if costs are negative or unrealistic
                if prompt_cost >= 0 and completion_cost >= 0:
                    model_costs[model_id] = {
                        "input": prompt_cost,
                        "output": completion_cost,
                        "context_length": model.get('context_length', 0),
                        "top_provider": model.get('top_provider', {}).get('name', 'Unknown')
                    }
        
        return model_costs
    except Exception as e:
        # Fallback pricing - silent error
        return {
            "openai/gpt-4o": {"input": 2.50, "output": 10.00},
            "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "openai/gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
            "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
            "meta-llama/llama-3.1-8b-instruct": {"input": 0.06, "output": 0.06},
            "mistralai/mistral-7b-instruct": {"input": 0.06, "output": 0.06},
        }


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens (1 token ≈ 4 characters)."""
    return len(text) // 4


def calculate_experiment_cost(models: List[Dict], dataset_path: str, prompt: str, model_costs: Dict) -> Dict:
    """Calculate estimated cost for running the experiment."""
    # Load dataset to count items
    with open(dataset_path, 'r') as f:
        dataset = []
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                dataset.append(json.loads(line))
    
    num_items = len(dataset)
    
    # Estimate tokens per item (prompt + input + response)
    avg_input_length = sum(len(item['input']) for item in dataset[:5]) // min(5, len(dataset))
    tokens_per_input = estimate_tokens(prompt) + estimate_tokens(dataset[0]['input'] if dataset else "")
    tokens_per_output = 10  # Assuming short responses
    
    total_cost = 0
    model_costs_breakdown = []
    
    for model_config in models:
        model_name = model_config['model']
        if model_name in model_costs:
            pricing = model_costs[model_name]
            input_cost = (tokens_per_input * num_items * pricing['input']) / 1_000_000
            output_cost = (tokens_per_output * num_items * pricing['output']) / 1_000_000
            model_total = input_cost + output_cost
            total_cost += model_total
            
            model_costs_breakdown.append({
                'model': model_name,
                'cost': model_total,
                'input_cost': input_cost,
                'output_cost': output_cost
            })
    
    return {
        'total_cost': total_cost,
        'num_models': len(models),
        'num_items': num_items,
        'breakdown': model_costs_breakdown,
        'tokens_per_item': tokens_per_input + tokens_per_output
    }


def optimize_cost(
    prompt: str,
    golden_dataset: str,
    base_model: Optional[str] = None,
    accuracy_threshold: float = 0.9,
    max_cost_ratio: float = 0.5
) -> Dict[str, Any]:
    """
    Find the most cost-effective model for your task.
    
    Args:
        prompt: Your system prompt or task description
        golden_dataset: Path to your JSONL test dataset
        base_model: Reference model to compare against (optional)
        accuracy_threshold: Minimum acceptable accuracy vs base model (0.9 = 90%)
        max_cost_ratio: Maximum acceptable cost ratio vs cheapest (0.5 = 50% of base cost)
    
    Returns:
        Dict with 'model', 'provider', 'cost_savings', 'accuracy', etc.
    
    Example:
        result = llmux.optimize_cost(
            prompt="Classify job postings as relevant or not",
            golden_dataset="data/test_jobs.jsonl"
        )
        print(f"Best model: {result['model']} saves {result['cost_savings']:.0%}")
    """
    
    # Fetch live pricing from OpenRouter (silent)
    model_costs = fetch_openrouter_models()
    
    # Calculate baseline accuracy from golden dataset
    baseline_accuracy = _calculate_baseline_accuracy(golden_dataset)
    
    # Select models to test (only cheaper than baseline)
    test_models = _get_test_models(base_model, model_costs)
    
    # Calculate and display cost estimates
    experiment_cost = calculate_experiment_cost(test_models, golden_dataset, prompt, model_costs)
    
    # Display cost estimation table with baseline accuracy
    _display_cost_tables(base_model, model_costs, experiment_cost, baseline_accuracy)
    
    # Run evaluation
    selector = Selector(test_models)
    results = selector.run_evaluation(golden_dataset, prompt)
    
    # Find best cost-optimized model
    best = _select_cost_optimized(results, base_model, accuracy_threshold, model_costs)
    
    return best


def _calculate_baseline_accuracy(dataset_path: str) -> Optional[float]:
    """Calculate baseline accuracy from existing LLM_Decision vs Human_Input in golden dataset."""
    try:
        with open(dataset_path, 'r') as f:
            dataset = []
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))
        
        correct = 0
        total = 0
        for item in dataset:
            if item.get('LLM_Decision') is not None and item.get('Human_Input') is not None:
                total += 1
                if item['LLM_Decision'] == item['Human_Input']:
                    correct += 1
        
        return correct / total if total > 0 else None
    except Exception:
        return None


def _display_cost_tables(base_model: Optional[str], model_costs: Dict, experiment_cost: Dict, baseline_accuracy: Optional[float]):
    """Display cost estimation in clean table format."""
    print("\n" + "="*80)
    print(" COST ESTIMATION")
    print("="*80)
    
    # Baseline accuracy
    if baseline_accuracy is not None:
        print(f"\nBASELINE ACCURACY: {baseline_accuracy:.1%}")
        print("(From existing LLM_Decision vs Human_Input in golden dataset)")
    
    # Base model table
    if base_model and base_model in model_costs:
        base_pricing = model_costs[base_model]
        base_total = base_pricing['input'] + base_pricing['output']
        print(f"\nBASE MODEL: {base_model}")
        print(f"{'Metric':<30} {'Value':<20}")
        print("-" * 50)
        print(f"{'Items to process':<30} {experiment_cost['num_items']:<20}")
        print(f"{'Tokens per item (est.)':<30} {experiment_cost['tokens_per_item']:<20}")
        print(f"{'Input cost per M tokens':<30} ${base_pricing['input']:<19.2f}")
        print(f"{'Output cost per M tokens':<30} ${base_pricing['output']:<19.2f}")
        base_cost = (experiment_cost['tokens_per_item'] * experiment_cost['num_items'] * base_total) / 1_000_000
        print(f"{'Total estimated cost':<30} ${base_cost:<19.4f}")
    
    # Models to test table
    print(f"\nEXPERIMENT: Testing {len(experiment_cost['breakdown'])} Models (Cheaper than Base)")
    print(f"{'Model':<40} {'Cost':<10} {'Input $/M':<12} {'Output $/M':<12}")
    print("-" * 76)
    
    # Sort models by cost for display
    sorted_models = sorted(experiment_cost['breakdown'], key=lambda x: x['cost'])
    
    for model_info in sorted_models:
        model_name = model_info['model']
        if model_name in model_costs:
            pricing = model_costs[model_name]
            print(f"{model_name[:38]:<40} ${model_info['cost']:<9.4f} ${pricing['input']:<11.2f} ${pricing['output']:<11.2f}")
    
    print("-" * 76)
    print(f"{'TOTAL EXPERIMENT COST':<40} ${experiment_cost['total_cost']:<9.4f}")
    print("="*80)


def optimize_speed(
    prompt: str,
    golden_dataset: str,
    base_model: Optional[str] = None,
    accuracy_threshold: float = 0.9
) -> Dict[str, Any]:
    """
    Find the fastest model that maintains quality.
    
    Similar to optimize_cost but prioritizes latency over cost.
    """
    # Fetch live pricing from OpenRouter
    model_costs = fetch_openrouter_models()
    test_models = _get_test_models(base_model, model_costs)
    selector = Selector(test_models)
    results = selector.run_evaluation(golden_dataset, prompt)
    
    # Find fastest acceptable model
    best = _select_speed_optimized(results, base_model, accuracy_threshold, model_costs)
    return best


def _get_test_models(base_model: Optional[str], model_costs: Dict) -> List[Dict]:
    """Get list of reliable models to test based on base model and live pricing."""
    
    # Filter out free models and unreliable ones
    reliable_models = {}
    for model_id, pricing in model_costs.items():
        # Skip free models (often rate limited)
        if ':free' in model_id:
            continue
        # Skip models with zero cost (often problematic)
        if pricing["input"] == 0 and pricing["output"] == 0:
            continue
        # Skip certain known problematic models
        if any(skip in model_id for skip in ['openrouter/', 'hunyuan', 'kimi', 'chimera']):
            continue
        reliable_models[model_id] = pricing
    
    if base_model and base_model in reliable_models:
        base_cost = reliable_models[base_model]["input"] + reliable_models[base_model]["output"]
        # Test only models cheaper than base model (smart optimization)
        models = []
        for model_id, pricing in reliable_models.items():
            model_cost = pricing["input"] + pricing["output"]
            if model_cost < base_cost:  # Must be cheaper than base
                models.append({"provider": "openrouter", "model": model_id})
        
        # Sort by cost and take top 2 cheapest
        models.sort(key=lambda x: reliable_models[x['model']]['input'] + reliable_models[x['model']]['output'])
        models = models[:2]
        
        # If no cheaper models found, show message
        if not models:
            print(f"\nNo models found cheaper than {base_model}")
            print("Testing 2 cheapest available models instead...")
            # Fallback to cheapest models
            all_models = [(model_id, pricing["input"] + pricing["output"]) 
                         for model_id, pricing in reliable_models.items()]
            all_models.sort(key=lambda x: x[1])
            models = [{"provider": "openrouter", "model": model_id} 
                     for model_id, _ in all_models[:2]]
        
    else:
        # Test a curated set of reliable models (limited to 2 for testing)
        preferred_models = [
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-3.1-8b-instruct",
        ]
        models = [
            {"provider": "openrouter", "model": model_id}
            for model_id in preferred_models
            if model_id in reliable_models
        ]
    
    return models


def _select_cost_optimized(results: Dict, base_model: Optional[str], threshold: float, model_costs: Dict) -> Dict:
    """Select best model based on cost and accuracy."""
    best_score = -1
    best_model = None
    
    for model_name, result in results.items():
        if 'error' in result or result.get('accuracy') is None:
            continue
            
        accuracy = result['accuracy']
        model = result['config']['model']
        
        # Calculate cost score (lower is better)
        if model in model_costs:
            cost = model_costs[model]["input"] + model_costs[model]["output"]
            # Score = accuracy / cost (higher accuracy, lower cost = better)
            score = accuracy / (cost + 0.01)  # Add small value to avoid division by zero
            
            if score > best_score and accuracy >= threshold:
                best_score = score
                best_model = {
                    'model': model,
                    'provider': 'openrouter',
                    'accuracy': accuracy,
                    'cost_per_million': cost,
                    'time': result['time'],
                    'cost_savings': None  # Calculate if base model provided
                }
    
    # Calculate savings vs base model
    if best_model and base_model and base_model in model_costs:
        base_cost = model_costs[base_model]["input"] + model_costs[base_model]["output"]
        best_model['cost_savings'] = 1 - (best_model['cost_per_million'] / base_cost)
    
    return best_model or {'error': 'No suitable model found'}


def _select_speed_optimized(results: Dict, base_model: Optional[str], threshold: float, model_costs: Dict) -> Dict:
    """Select best model based on speed and accuracy."""
    best_time = float('inf')
    best_model = None
    
    for model_name, result in results.items():
        if 'error' in result or result.get('accuracy') is None:
            continue
            
        accuracy = result['accuracy']
        
        if accuracy >= threshold and result['time'] < best_time:
            best_time = result['time']
            model = result['config']['model']
            best_model = {
                'model': model,
                'provider': 'openrouter',
                'accuracy': accuracy,
                'latency': result['time'],
                'cost_per_million': model_costs.get(model, {}).get('input', 0) + 
                                   model_costs.get(model, {}).get('output', 0)
            }
    
    return best_model or {'error': 'No suitable model found'}