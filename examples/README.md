# Examples

This folder contains example scripts demonstrating how to use LLMuxer.

## Examples

### simple_usage.py
Basic example showing the simplest way to use LLMuxer with a pre-existing dataset.

```bash
python examples/simple_usage.py
```

### classification_demo.py
Demonstrates classification tasks with sentiment analysis using the Banking77 dataset.

```bash
python examples/classification_demo.py
```

### banking77_test.py
Complete example for banking intent classification using the Banking77 dataset.

```bash
# First prepare the data
python scripts/prepare_banking77.py

# Then run the test
python examples/banking77_test.py
```

## Data

The `data/` subfolder in examples can be used for storing sample datasets for testing.

## Getting Started

1. Set your OpenRouter API key:
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   ```

2. Install LLMuxer:
   ```bash
   pip install llmuxer
   ```

3. Run any example:
   ```bash
   python examples/simple_usage.py
   ```