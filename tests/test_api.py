"""Test API functionality."""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock
from llmuxer.api import (
    optimize_cost,
    _prepare_dataset,
    _detect_task_and_options,
    estimate_tokens,
    calculate_experiment_cost
)


class TestOptimizeCost:
    """Test the main optimize_cost function."""
    
    def test_basic_classification(self):
        """Test basic classification task."""
        examples = [
            {"input": "Great product!", "label": "positive"},
            {"input": "Terrible!", "label": "negative"}
        ]
        
        with patch('llmuxer.api.fetch_openrouter_models') as mock_fetch:
            mock_fetch.return_value = {
                "openai/gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
            }
            
            with patch('llmuxer.selector.Selector.run_evaluation') as mock_eval:
                mock_eval.return_value = {
                    "openai/gpt-3.5-turbo": {
                        "accuracy": 0.95,
                        "time": 2.0,
                        "config": {"provider": "openrouter", "model": "openai/gpt-3.5-turbo"}
                    }
                }
                
                result = optimize_cost(
                    baseline="gpt-4",
                    examples=examples,
                    task="classification",
                    options=["positive", "negative"],
                    min_accuracy=0.9
                )
                
                assert "model" in result
                assert result["accuracy"] >= 0.9
    
    def test_no_suitable_model(self):
        """Test when no model meets accuracy threshold."""
        examples = [{"input": "test", "label": "A"}]
        
        with patch('llmuxer.api.fetch_openrouter_models') as mock_fetch:
            mock_fetch.return_value = {
                "openai/gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
            }
            
            with patch('llmuxer.selector.Selector.run_evaluation') as mock_eval:
                mock_eval.return_value = {
                    "openai/gpt-3.5-turbo": {
                        "accuracy": 0.5,  # Below threshold
                        "time": 2.0,
                        "config": {"provider": "openrouter", "model": "openai/gpt-3.5-turbo"}
                    }
                }
                
                result = optimize_cost(
                    baseline="gpt-4",
                    examples=examples,
                    min_accuracy=0.9
                )
                
                assert "error" in result or result.get("accuracy", 0) < 0.9
    
    def test_sample_size(self):
        """Test sample_size parameter."""
        # Create larger dataset
        examples = [{"input": f"text{i}", "label": f"label{i%2}"} for i in range(100)]
        
        with patch('llmuxer.api.fetch_openrouter_models') as mock_fetch:
            mock_fetch.return_value = {"test/model": {"input": 0.1, "output": 0.1}}
            
            with patch('llmuxer.selector.Selector.run_evaluation') as mock_eval:
                mock_eval.return_value = {
                    "test/model": {"accuracy": 0.95, "time": 1.0, "config": {}}
                }
                
                with patch('llmuxer.api._prepare_dataset') as mock_prep:
                    mock_prep.return_value = ("temp.jsonl", examples[:20])  # Should sample 20%
                    
                    optimize_cost(
                        baseline="gpt-4",
                        examples=examples,
                        sample_size=0.2
                    )
                    
                    # Check that sample_size was passed
                    mock_prep.assert_called()
                    call_args = mock_prep.call_args[1]
                    assert call_args.get('sample_size') == 0.2


class TestPrepareDataset:
    """Test dataset preparation."""
    
    def test_examples_to_jsonl(self):
        """Test converting examples to JSONL format."""
        examples = [
            {"input": "text1", "label": "A"},
            {"input": "text2", "label": "B"}
        ]
        
        temp_path, processed = _prepare_dataset(examples=examples)
        
        try:
            # Check file was created
            assert os.path.exists(temp_path)
            
            # Check content
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2
                
                first = json.loads(lines[0])
                assert first["input"] == "text1"
                assert first["label"] == "A"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_dataset_file_loading(self):
        """Test loading from JSONL file."""
        # Create temp dataset
        dataset = [
            {"input": "test1", "label": "A"},
            {"input": "test2", "label": "B"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            result_path, processed = _prepare_dataset(dataset=temp_path)
            
            assert len(processed) == 2
            assert processed[0]["input"] == "test1"
            assert processed[0]["label"] == "A"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(result_path):
                os.unlink(result_path)
    
    def test_backward_compatibility_ground_truth(self):
        """Test that ground_truth field still works."""
        examples = [
            {"input": "text", "ground_truth": "label"}  # Old format
        ]
        
        temp_path, processed = _prepare_dataset(examples=examples)
        
        try:
            with open(temp_path, 'r') as f:
                data = json.loads(f.readline())
                assert data["label"] == "label"  # Converted to label
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDetectTaskAndOptions:
    """Test task and options detection."""
    
    def test_classification_detection(self):
        """Test detecting classification task."""
        examples = [
            {"input": "a", "label": "A"},
            {"input": "b", "label": "B"},
            {"input": "c", "label": "A"}
        ]
        
        task, options = _detect_task_and_options(examples)
        
        assert task == "classification"
        assert set(options) == {"A", "B"}
    
    def test_binary_detection(self):
        """Test detecting binary task."""
        examples = [
            {"input": "a", "label": "yes"},
            {"input": "b", "label": "no"},
            {"input": "c", "label": "yes"}
        ]
        
        task, options = _detect_task_and_options(examples)
        
        # Should detect as classification with 2 options
        assert task == "classification"
        assert len(options) == 2
    
    def test_empty_dataset(self):
        """Test with empty dataset."""
        task, options = _detect_task_and_options([])
        
        assert task is None
        assert options is None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        # Roughly 1 token per 4 characters
        assert estimate_tokens("1234") == 1
        assert estimate_tokens("12345678") == 2
        assert estimate_tokens("") == 0
    
    def test_calculate_experiment_cost(self):
        """Test cost calculation."""
        # Create temp dataset
        dataset = [
            {"input": "test" * 10, "label": "A"}  # 40 chars = ~10 tokens
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            models = [{"model": "test/model"}]
            model_costs = {
                "test/model": {"input": 0.001, "output": 0.002}
            }
            
            result = calculate_experiment_cost(
                models=models,
                dataset_path=temp_path,
                prompt="Test prompt",
                model_costs=model_costs
            )
            
            assert "total_cost" in result
            assert "model_costs" in result
            assert result["total_cost"] > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])