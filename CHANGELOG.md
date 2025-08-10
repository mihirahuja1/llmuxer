# Changelog

All notable changes to LLMuxer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial development version

## [0.1.0] - 2024-01-XX

### Added
- Initial release of LLMuxer
- Multi-provider model testing across 18 models from 7 providers:
  - OpenAI (GPT-4o-mini, GPT-3.5-turbo)
  - Anthropic (Claude-3-haiku, Claude-3-sonnet)
  - Google (Gemini-1.5-flash-8b, Gemini-1.5-flash, Gemini-1.5-pro)
  - Qwen (2.5-7b, 2.5-14b, 2.5-72b)
  - DeepSeek (chat, coder, v2.5)
  - Mistral (7b, Mixtral-8x7b, large)
  - Meta (Llama-3.1-8b, Llama-3.1-70b)
- Classification task support (sentiment analysis, intent detection, categorization)
- Smart stopping algorithm for model families
- Cost comparison and savings calculation
- Sample size control for faster testing (0-100% of dataset)
- JSONL dataset format support
- Baseline model comparison
- Detailed cost breakdown showing:
  - Input/output token counts
  - Cost per model
  - Total experiment cost
  - Percentage savings vs baseline
- OpenRouter API integration for unified model access
- Examples for common use cases:
  - Sentiment analysis
  - Banking intent classification (Banking77 dataset)

### Features
- `optimize_cost()` main API function
- Automatic model selection based on accuracy threshold
- Token estimation and cost calculation
- Progress tracking during evaluation
- Error handling and retry logic

### Known Limitations
- Only classification tasks supported (extraction, generation, binary coming in v0.2.0)
- Synchronous evaluation only (async coming in v0.2.0)
- No caching of evaluations yet
- Limited to accuracy metric (F1, precision, recall coming soon)

[0.1.0]: https://github.com/mihirahuja/llmuxer/releases/tag/v0.1.0