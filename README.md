# LLMux

Cut your LLM costs by 90% with one line of code. Automatically find the cheapest model that maintains quality for your specific task.

## Installation

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key-here"
```

## Quick Start

```python
import llmux

# One line to optimize your LLM costs
result = llmux.optimize_cost(
    prompt="Your task description here",
    golden_dataset="your_test_data.jsonl"
)

print(f"Best model: {result['model']}")
print(f"Cost savings: {result['cost_savings']:.0%}")
```

That's it! LLMux will:
1. Test multiple models from cheap to expensive
2. Measure accuracy on your golden dataset
3. Return the cheapest model that meets your quality bar

## API

### `optimize_cost()`
Find the most cost-effective model for your task.

```python
result = llmux.optimize_cost(
    prompt="Classify customer sentiment",
    golden_dataset="data/test.jsonl",
    base_model="openai/gpt-4o",         # Optional: compare against this model
    accuracy_threshold=0.9               # Optional: minimum acceptable accuracy
)
```

### `optimize_speed()`
Find the fastest model that maintains quality.

```python
result = llmux.optimize_speed(
    prompt="Extract entities from text",
    golden_dataset="data/test.jsonl"
)
```

## Data Format

Your test data should be JSONL with:
```json
{"input": "your input text", "LLM_Decision": null, "Human_Input": 1}
```

- `input`: The input for your task
- `Human_Input`: Ground truth label (optional for initial testing)
- `LLM_Decision`: Filled automatically during evaluation

## Example

See `simple_usage.py` for a complete example.

## Supported Models

LLMux tests models from OpenRouter including:
- GPT-3.5, GPT-4, GPT-4o Mini
- Claude 3 Haiku, Sonnet, Opus
- Gemini Flash, Pro
- Llama 3.1 (8B, 70B)
- Mistral, Mixtral
- And more...
