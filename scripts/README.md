# Utility Scripts

This folder contains utility scripts for data preparation and testing.

## Scripts

### prepare_banking77.py
Downloads and prepares the Banking77 dataset for testing intent classification.

```bash
python scripts/prepare_banking77.py
```

Creates `data/banking77_test.jsonl` with 100 samples.

### populate_baseline.py
Adds baseline model predictions to an existing dataset for comparison.

```bash
python scripts/populate_baseline.py
```

### list_models.py
Lists all available models and their pricing from OpenRouter API.

```bash
python scripts/list_models.py
```

Requires `OPENROUTER_API_KEY` environment variable.