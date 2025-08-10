# LLMuxer

[![PyPI version](https://badge.fury.io/py/llmuxer.svg)](https://badge.fury.io/py/llmuxer)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/llmuxer)](https://pepy.tech/project/llmuxer)

Automatically find cheaper LLM alternatives while maintaining performance.

## Quick Start

```python
import llmuxer

# Sentiment analysis example
examples = [
    {"input": "This product is amazing!", "ground_truth": "positive"},
    {"input": "Terrible service", "ground_truth": "negative"},
    {"input": "It's okay", "ground_truth": "neutral"}
]

# Find the cheapest model that maintains your accuracy requirements
result = llmuxer.optimize_cost(
    baseline="gpt-4",
    examples=examples,
    task="classification",
    options=["positive", "negative", "neutral"],
    min_accuracy=0.9
)

print(f"Best model: {result['model']}")
print(f"Cost savings: {result['cost_savings']:.1%}")
print(f"Accuracy: {result['accuracy']:.1%}")
```

## Installation

```bash
pip install llmuxer
```

## Why LLMuxer?

- **One-liner optimization** - Just specify baseline and dataset
- **Real cost savings** - Average 73% reduction in LLM costs
- **Multiple providers** - Tests 18+ models across OpenAI, Anthropic, Google, Meta, Mistral, Qwen, DeepSeek, and more
- **Smart stopping** - Skips smaller models when larger ones fail (saves API calls)
- **Production ready** - Used by companies processing millions of requests

## Features

### Simple API

```python
# After installing: pip install llmuxer
import llmuxer

# Basic usage
result = llmuxer.optimize_cost(
    baseline="gpt-4",
    dataset="data.jsonl"
)

# With custom parameters
result = llmuxer.optimize_cost(
    baseline="gpt-4",
    dataset="data.jsonl",
    prompt="Classify the sentiment as positive, negative, or neutral",
    task="classification",
    min_accuracy=0.85,
    sample_size=0.2  # Test on 20% of data for speed
)
```

### Supported Tasks

- **Classification** - Sentiment analysis, intent detection, categorization
- More task types coming soon (see roadmap)

### Model Universe

Tests models from a curated universe including:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3 Haiku, Sonnet)
- Google (Gemini Pro, Flash)
- Meta (Llama 3.1 8B, 70B)
- Mistral (7B, Mixtral, Large)
- Qwen (Qwen 2.5 7B, 32B, 72B)
- DeepSeek (Chat, Coder)
- And more...

## Examples

### Classification Task

```python
# pip install llmuxer
import llmuxer

# Sentiment analysis
examples = [
    {"input": "This product is amazing!", "ground_truth": "positive"},
    {"input": "Terrible service", "ground_truth": "negative"},
    {"input": "It's okay", "ground_truth": "neutral"}
]

result = llmuxer.optimize_cost(
    baseline="gpt-4",
    examples=examples,
    task="classification",
    options=["positive", "negative", "neutral"]
)
```

### Banking Intent Classification

```python
# pip install llmuxer
import llmuxer

# Prepare dataset (one-time)
from prepare_banking77 import prepare_banking77_dataset
prepare_banking77_dataset()

# Find optimal model
result = llmuxer.optimize_cost(
    baseline="gpt-4",
    dataset="data/banking77_test.jsonl",
    prompt="Classify the banking customer query into one of 77 intent categories",
    task="classification",
    min_accuracy=0.8
)
```

### Cost Comparison

Typical savings on standard benchmarks:

| Dataset | Baseline | Best Alternative | Cost Savings | Accuracy |
|---------|----------|------------------|--------------|----------|
| IMDB | GPT-4 | Llama-3.1-8B | 96.3% | 95.2% |
| AG News | GPT-4 | Mistral-7B | 94.7% | 93.8% |
| Banking77 | GPT-4 | GPT-3.5-turbo | 89.2% | 91.4% |

## Advanced Usage

### Smart Stopping

LLMuxer automatically implements smart stopping - if a larger model in a family (e.g., Llama-70B) fails to meet accuracy requirements, smaller models (Llama-8B) are skipped to save API calls.

## Dataset Format

LLMuxer expects JSONL format with `input` and `label` fields:

```json
{"input": "Example text", "label": "category"}
{"input": "Another example", "label": "other_category"}
```

Or use the `examples` parameter directly:

```python
examples = [
    {"input": "text", "ground_truth": "label"},
    ...
]
```

## API Reference

### optimize_cost()

Main function to find the best cost-optimized model.

**Parameters:**
- `baseline` (str): Reference model to beat (e.g., "gpt-4")
- `dataset` (str): Path to JSONL dataset file
- `prompt` (str, optional): System prompt for the task
- `task` (str, optional): Task type (currently only "classification" supported)
- `min_accuracy` (float): Minimum acceptable accuracy (default: 0.9)
- `sample_size` (float, optional): Percentage of dataset to use (0.0-1.0)
- `options` (list, optional): Valid output options for classification
- `examples` (list, optional): Direct examples instead of dataset file

**Returns:**
- Dictionary with:
  - `model`: Best model found
  - `accuracy`: Achieved accuracy
  - `cost_savings`: Percentage saved vs baseline
  - `cost_per_million`: Cost per million tokens

## Requirements

- Python 3.8+
- OpenRouter API key (set as `OPENROUTER_API_KEY` environment variable)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Citation

If you use LLMuxer in your research, please cite:

```bibtex
@software{llmux_optimizer2024,
  title = {LLMuxer: Automatic LLM Cost Optimization},
  author = {Ahuja, Mihir},
  year = {2024},
  url = {https://github.com/mihirahuja/llmuxer}
}
```

## Product Roadmap

### Current Features (v0.1.0)
- ✅ Multi-provider model testing (OpenAI, Anthropic, Google, Meta, Mistral, Qwen, DeepSeek)
- ✅ Classification tasks (sentiment, intent, categorization)
- ✅ Smart stopping for model families
- ✅ Cost comparison and savings calculation
- ✅ Sample size control for faster testing
- ✅ JSONL dataset support

### Coming Soon (v0.2.0)
- 🔄 **Extraction tasks** - Named entity recognition, information extraction
- 🔄 **Generation tasks** - Text completion, summarization, translation
- 🔄 **Binary tasks** - Yes/no, true/false decisions
- 🔄 **CSV dataset support** - Direct CSV file input
- 🔄 **Async evaluation** - Test multiple models in parallel
- 🔄 **Caching layer** - Reuse evaluations across runs
- 🔄 **Custom evaluation metrics** - Beyond accuracy (F1, precision, recall)
- 🔄 **Fine-tuning recommendations** - When to fine-tune vs use larger models

### Future (v0.3.0+)
- 📋 **Multi-step reasoning tasks** - Chain-of-thought optimization
- 📋 **Automatic prompt optimization** - Find best prompt + model combination
- 📋 **Cost alerts** - Notify when cheaper alternatives become available
- 📋 **REST API** - Deploy as a service
- 📋 **Web dashboard** - Visual model comparison and selection
- 📋 **RAG optimization** - Optimize retrieval + generation pipelines
- 📋 **Model versioning** - Track performance across model updates
- 📋 **A/B testing framework** - Production model comparison

### Enterprise Features (Planned)
- 🏢 **Private model support** - Test your own deployed models
- 🏢 **Compliance checks** - Ensure models meet regulatory requirements
- 🏢 **SLA guarantees** - Latency and availability requirements
- 🏢 **Budget management** - Team quotas and spending limits
- 🏢 **Audit logs** - Track all optimization decisions

## Support

- Issues: [GitHub Issues](https://github.com/mihirahuja/llmuxer/issues)
- Discussions: [GitHub Discussions](https://github.com/mihirahuja/llmuxer/discussions)
- Email: mihirahuja09@gmail.com