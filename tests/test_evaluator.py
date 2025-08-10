"""Test evaluator functionality."""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock
from llmuxer.evaluator import Evaluator


class TestEvaluator:
    """Test the Evaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_provider = Mock()
        self.mock_provider.send_message = MagicMock(return_value="positive")
        self.evaluator = Evaluator(self.mock_provider)
    
    def test_evaluate_single(self):
        """Test evaluating a single example."""
        self.evaluator.evaluate_single(
            input_text="Great product!",
            options=["positive", "negative"]
        )
        
        # Check that provider was called
        self.mock_provider.send_message.assert_called_once()
        call_args = self.mock_provider.send_message.call_args[1]
        assert "Great product!" in call_args["user_message"]
    
    def test_evaluate_dataset(self):
        """Test evaluating a full dataset."""
        # Create temp dataset
        dataset = [
            {"input": "Great!", "label": "positive", "LLM_Decision": None},
            {"input": "Bad!", "label": "negative", "LLM_Decision": None}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            # Mock provider responses
            self.mock_provider.send_message.side_effect = ["positive", "negative"]
            
            accuracy, results = self.evaluator.evaluate(
                dataset=temp_path,
                system_prompt="Classify sentiment",
                options=["positive", "negative"]
            )
            
            assert accuracy == 1.0  # Both correct
            assert len(results) == 2
            assert results[0]["LLM_Decision"] == "positive"
            assert results[1]["LLM_Decision"] == "negative"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_evaluate_with_errors(self):
        """Test evaluation with some errors."""
        dataset = [
            {"input": "test1", "label": "A", "LLM_Decision": None},
            {"input": "test2", "label": "B", "LLM_Decision": None}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            # Mock one success, one failure
            self.mock_provider.send_message.side_effect = ["A", Exception("API Error")]
            
            accuracy, results = self.evaluator.evaluate(
                dataset=temp_path,
                options=["A", "B"]
            )
            
            # Should handle error gracefully
            assert accuracy <= 1.0
            assert len(results) == 2
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        dataset = [
            {"label": "A", "LLM_Decision": "A"},  # Correct
            {"label": "B", "LLM_Decision": "B"},  # Correct
            {"label": "A", "LLM_Decision": "B"},  # Wrong
            {"label": "C", "LLM_Decision": None}  # No prediction
        ]
        
        accuracy = self.evaluator.calculate_accuracy(dataset)
        
        # 2 correct out of 3 valid predictions
        assert accuracy == pytest.approx(2/3, rel=1e-3)
    
    def test_normalize_output(self):
        """Test output normalization."""
        options = ["positive", "negative", "neutral"]
        
        # Exact match
        assert self.evaluator.normalize_output("positive", options) == "positive"
        
        # Case insensitive
        assert self.evaluator.normalize_output("POSITIVE", options) == "positive"
        
        # With extra spaces
        assert self.evaluator.normalize_output("  positive  ", options) == "positive"
        
        # Partial match
        assert self.evaluator.normalize_output("It's positive!", options) == "positive"
        
        # No match
        assert self.evaluator.normalize_output("unknown", options) == "unknown"
    
    def test_format_options_string(self):
        """Test options formatting for prompt."""
        options = ["A", "B", "C"]
        formatted = self.evaluator.format_options(options)
        
        assert "A" in formatted
        assert "B" in formatted
        assert "C" in formatted
        assert formatted.startswith("[") and formatted.endswith("]")
    
    def test_backward_compatibility_human_input(self):
        """Test that old Human_Input field still works."""
        dataset = [
            {"input": "test", "Human_Input": "A", "LLM_Decision": "A"}  # Old format
        ]
        
        accuracy = self.evaluator.calculate_accuracy(dataset)
        assert accuracy == 1.0  # Should recognize Human_Input as label


if __name__ == "__main__":
    pytest.main([__file__, "-v"])