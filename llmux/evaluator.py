"""Evaluator module for testing model performance."""

import json
import time
import threading
from typing import List, Dict, Any, Tuple
from .provider import Provider


class Evaluator:
    """Evaluates models against golden dataset."""
    
    def __init__(self, provider: Provider):
        self.provider = provider
        self._stop_spinner = False
        
    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        """Load JSONL dataset."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
        return data
    
    def _spinner_animation(self, message: str):
        """Animated spinner with green color."""
        frames = ["✦", "✧", "★", "✱", "✴", "✱", "★", "✧", "✦"]
        GREEN = "\033[92m"
        RESET = "\033[0m"
        
        i = 0
        while not self._stop_spinner:
            frame = frames[i % len(frames)]
            print(f"\r   {GREEN}{frame}{RESET} {message}", end="", flush=True)
            time.sleep(0.1)
            i += 1
    
    def evaluate(self, dataset: List[Dict[str, Any]], system_prompt: str = "") -> Tuple[float, List[Dict]]:
        """Evaluate model on dataset, returns accuracy and updated results.
        
        Updates each item with LLM_Decision field.
        If Human_Input is provided, calculates accuracy against it.
        """
        correct = 0
        total_with_labels = 0
        results = []
        
        # Start spinner in background thread
        self._stop_spinner = False
        spinner_thread = threading.Thread(
            target=self._spinner_animation, 
            args=(f"Processing {len(dataset)} items...",)
        )
        spinner_thread.daemon = True
        spinner_thread.start()
        
        for idx, item in enumerate(dataset):            
            # Build prompt with the input field
            full_prompt = f"{system_prompt}\n\n{item['input']}\n\nRespond with only '1' to Apply or '0' to Skip."
            response = self.provider.complete(full_prompt).strip()
            
            # Extract 0 or 1 from response
            prediction = 1 if '1' in response else 0
            
            # Update the item with LLM decision
            item_result = item.copy()
            item_result['LLM_Decision'] = prediction
            
            # If Human_Input exists and is not null, check accuracy
            if item.get('Human_Input') is not None:
                total_with_labels += 1
                if prediction == item['Human_Input']:
                    correct += 1
            
            results.append(item_result)
        
        # Stop spinner
        self._stop_spinner = True
        if spinner_thread.is_alive():
            spinner_thread.join(timeout=0.2)
        print("\r   ", end="")  # Clear spinner line
                
        # Calculate accuracy only if we have labeled data
        accuracy = correct / total_with_labels if total_with_labels > 0 else None
        
        # Calculate overall accuracy from LLM_Decision vs Human_Input
        llm_correct = 0
        total_decisions = 0
        for result in results:
            if result.get('LLM_Decision') is not None and result.get('Human_Input') is not None:
                total_decisions += 1
                if result['LLM_Decision'] == result['Human_Input']:
                    llm_correct += 1
        
        overall_accuracy = llm_correct / total_decisions if total_decisions > 0 else None
        
        return overall_accuracy, results