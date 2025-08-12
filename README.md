# LLMuxer

<p align="center">
  <img src="assets/muxerlogo.png" alt="LLMuxer Logo" width="400">
</p>

[![PyPI version](https://badge.fury.io/py/llmuxer.svg)](https://pypi.org/project/llmuxer/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Tests](https://github.com/mihirahuja1/llmuxer/workflows/CI/badge.svg)](https://github.com/mihirahuja1/llmuxer/actions)
[![Downloads](https://pepy.tech/badge/llmuxer)](https://pepy.tech/project/llmuxer)
[![GitHub Stars](https://img.shields.io/github/stars/mihirahuja1/llmuxer)](https://github.com/mihirahuja1/llmuxer/stargazers)

**Find the cheapest LLM that meets your quality bar** *(Currently supports classification tasks only)*

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mihirahuja1/llmuxer/blob/main/examples/quickstart.ipynb)

```python
import llmuxer

# Example: Classify sentiment with 90% accuracy requirement
examples = [
    {"input": "This product is amazing!", "label": "positive"},
    {"input": "Terrible service", "label": "negative"},
    {"input": "It's okay", "label": "neutral"}
]

result = llmuxer.optimize_cost(
    baseline="gpt-4o-mini",
    prompt="Classify the sentiment of this text as positive, negative, or neutral.",
    examples=examples,
    task="classification",  # Currently only classification is supported
    options=["positive", "negative", "neutral"],
    min_accuracy=0.9  # Require 90% accuracy
)

print(result)
# Takes ~30-60 seconds for small datasets, ~10-15 minutes for 1k samples
```

### Example Output
```python
{
    "model": "anthropic/claude-3-haiku",
    "accuracy": 0.92,
    "cost_per_million": 0.25,
    "cost_savings": 0.80,  # 80% cheaper than baseline
    "baseline_cost_per_million": 1.25,
    "tokens_evaluated": 1500
}
```

## The Problem

You're using an expensive model for classification. Could a cheaper model do just as well? What about Claude Haiku, Gemini Flash, or Llama models?

**LLMuxer automatically tests your classification task across 18 models to find the cheapest one that maintains your required accuracy.**

## How It Works

```
Your Dataset → LLMuxer → Tests 18 Models → Returns Cheapest That Works
                  ↓
           Uses OpenRouter API
           (unified interface)
```

LLMuxer:
1. Takes your baseline model (e.g., GPT-4) and test dataset
2. Evaluates cheaper alternatives via OpenRouter
3. Returns the cheapest model meeting your accuracy threshold
4. Shows detailed cost breakdown and savings

## Installation

### Prerequisites
- Python 3.8+
- [OpenRouter API key](https://openrouter.ai/keys) (for model access)

### Install
```bash
pip install llmuxer
```

### Setup
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Key Features

- **18 models tested** - OpenAI, Anthropic, Google, Meta, Mistral, Qwen, DeepSeek  
- **Smart stopping** - Skips smaller models if larger ones fail  
- **Cost breakdown** - See token counts and costs per model  
- **Fast testing** - Use `sample_size` to test on subset first  
- **Simple API** - One function does everything  
- **Classification only** - Support for extraction, generation, and binary tasks coming in v0.2
- **Accuracy evaluation** - Uses case-insensitive exact match against provided options ([see evaluator](llmuxer/evaluator.py#L164))

## Benchmarks

### Tested Models

*Live pricing data from OpenRouter API (updated automatically):*

| Provider | Models Count |
|----------|-------------|
| OpenAI | 2 models (gpt-4o-mini, gpt-3.5-turbo) |
| Anthropic | 2 models (claude-3-haiku, claude-3-sonnet) |
| Google | 3 models (gemini-1.5-pro, gemini-1.5-flash, gemini-1.5-flash-8b) |
| Qwen | 3 models (qwen-2.5-72b, qwen-2.5-14b, qwen-2.5-7b) |
| DeepSeek | 3 models (deepseek-v2.5, deepseek-coder, deepseek-chat) |
| Mistral | 3 models (mistral-large, mixtral-8x7b, mistral-7b) |
| Meta | 2 models (llama-3.1-70b, llama-3.1-8b) |

**Total: 18 models across 7 providers**

*See [Benchmarks](#benchmarks) for current pricing - prices vary by time and usage.*

### Reproduce Our Benchmarks

```bash
# Test all 18 models on Banking77 dataset (tested 2025-01-12)
python scripts/prepare_banking77.py
python examples/classification_demo.py
```

**Expected Results:** Most models achieve 85-92% accuracy on Banking77. Claude-3-haiku typically provides the best accuracy/cost ratio for classification tasks.

### Example Cost Calculation

**Sample Dataset** *(50 job classification samples)*

| Metric | Baseline Model | Optimized Model | Potential Savings |
|--------|---------------|-----------------|------------------|
| **Accuracy** | Baseline | 85%+ required | Quality threshold |
| **Cost/Million Tokens** | Varies | Lower cost | See benchmarks |
| **Tokens/Request** | ~150 (typical) | Same | Based on your data |

*Actual costs depend on current OpenRouter pricing and your specific use case.*

**[📊 Full Benchmark Report](docs/benchmarks.md)** | **[🔄 Reproduction Guide](#reproduction)**

### Reproduction

```bash
# Install and setup
pip install llmuxer
export OPENROUTER_API_KEY="your-key"

# Run exact benchmark  
./scripts/bench.sh

# Generates: benchmarks/bench_YYYYMMDD.json + docs/benchmarks.md
```

**Benchmark Notes:**
- Fixed dataset: `data/jobs_50.jsonl` (8 categories, 50 samples)
- Pinned models: 8 specific models with exact API versions  
- Conservative estimates: 150 tokens/request assumption
- No cherry-picking: Single test run results
- Quality threshold: 85%+ accuracy required

## API Reference

### `optimize_cost()`

Find the cheapest model meeting your requirements for classification tasks.

**Parameters:**
- `baseline` (str): Your current model (e.g., "gpt-4")
- `examples` (list): Test examples with input and label
- `dataset` (str): Path to JSONL file (alternative to examples)
- `task` (str): Must be "classification" (other tasks coming soon)
- `options` (list): Valid output classes for classification
- `min_accuracy` (float): Minimum acceptable accuracy (0.0-1.0)
- `sample_size` (float): Fraction of dataset to test (0.0-1.0)
- `prompt` (str): Optional system prompt

**Returns:**
Dictionary with model name, accuracy, cost, and savings.

**Error Handling:**
- Returns `{"error": "message"}` if no model meets threshold
- Retries on API failures
- Validates dataset format

## Full Example: Banking Intent Classification

```python
import llmuxer

# Using the Banking77 dataset (77 intent categories)
result = llmuxer.optimize_cost(
    baseline="gpt-4o-mini",
    dataset="data/banking77.jsonl",  # Your prepared dataset
    task="classification",
    min_accuracy=0.8,
    sample_size=0.2  # Test on 20% first for speed
)

if "error" in result:
    print(f"No model found: {result['error']}")
else:
    print(f"Switch from gpt-4o-mini to {result['model']}")
    print(f"Save {result['cost_savings']:.0%} on costs")
    print(f"Accuracy: {result['accuracy']:.1%}")
```

## Dataset Format

JSONL format with `input` and `label` fields:
```json
{"input": "What's my account balance?", "label": "balance_inquiry"}
{"input": "I lost my card", "label": "card_lost"}
```

## Performance Notes

### Timing Estimates

For a dataset with 1,000 samples:

| Phase | Estimated Time |
|-------|---------------|
| Per model evaluation | ~30-60 seconds |
| Smart stopping (skips failing families) | Saves 30-50% time |
| **Total for 18 models** | **~10-15 minutes** |

### Speed Considerations

- **Sequential Processing**: Currently tests one model at a time (parallel in v0.2)
- **Sample Size**: Use `sample_size=0.1` to test on 10% first for quick validation
- **Smart Stopping**: Saves 30-50% time by skipping smaller models when larger ones fail
- **Rate Limits**: Automatic handling with exponential backoff
- **Caching**: Not yet implemented (coming in v0.2 will reduce re-evaluation time by 90%)

## Links

- [PyPI Package](https://pypi.org/project/llmuxer/)
- [GitHub Repository](https://github.com/mihirahuja1/llmuxer)
- [OpenRouter API Keys](https://openrouter.ai/keys)
- [Full Documentation](https://github.com/mihirahuja1/llmuxer/tree/main/docs)
- [Roadmap](https://github.com/mihirahuja1/llmuxer/blob/main/ROADMAP.md)

## License

MIT - see [LICENSE](LICENSE) file.

## Support

- Issues: [GitHub Issues](https://github.com/mihirahuja1/llmuxer/issues)
- Email: mihirahuja09@gmail.com
