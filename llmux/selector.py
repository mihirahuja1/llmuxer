"""Selector module for choosing best performing model."""

import time
from typing import List, Dict, Any, Optional, Tuple
from .provider import Provider, get_provider
from .evaluator import Evaluator


class Selector:
    """Selects best model based on evaluation results."""
    
    def __init__(self, models: List[Dict[str, Any]]):
        """Initialize with list of model configs.
        
        Each config should have:
        - provider: 'openai' or 'anthropic'
        - model: model name
        - any other provider-specific kwargs
        """
        self.models = models
        self.results = {}
        
    def run_evaluation(self, dataset_path: str, system_prompt: str = "") -> Dict[str, Any]:
        """Run evaluation on all models."""
        print(f"\nStarting evaluation of {len(self.models)} models...")
        print("=" * 80)
        
        for i, config in enumerate(self.models, 1):
            # Don't modify original config
            config_copy = config.copy()
            provider_name = config_copy.pop('provider')
            model_name = config_copy.get('model', 'unknown')
            
            print(f"[{i}/{len(self.models)}] Testing {model_name}...")
            
            try:
                provider = get_provider(provider_name, **config_copy)
                evaluator = Evaluator(provider)
                dataset = evaluator.load_dataset(dataset_path)
                
                start_time = time.time()
                accuracy, results = evaluator.evaluate(dataset, system_prompt)
                elapsed = time.time() - start_time
                
                self.results[f"{provider_name}/{model_name}"] = {
                    'accuracy': accuracy,
                    'time': elapsed,
                    'results': results,
                    'config': {'provider': provider_name, **config_copy}
                }
                
                if accuracy is not None:
                    print(f"COMPLETED: {accuracy:.1%} accuracy in {elapsed:.2f}s")
                else:
                    print(f"COMPLETED in {elapsed:.2f}s (no ground truth labels)")
                print()
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    print(f"\r   SKIPPED: Rate limited")
                elif "404" in error_msg:
                    print(f"\r   SKIPPED: Model not found")
                else:
                    print(f"\r   ERROR: {error_msg}")
                print()
                self.results[f"{provider_name}/{model_name}"] = {
                    'error': str(e),
                    'config': {'provider': provider_name, **config_copy}
                }
        
        return self.results
    
    def get_best_model(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get the best performing model."""
        valid_results = {
            k: v for k, v in self.results.items() 
            if 'accuracy' in v and v['accuracy'] is not None
        }
        
        if not valid_results:
            return None
            
        best_model = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        return best_model