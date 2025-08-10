# LLMuxer Roadmap

## Current Version: 0.1.0

### ✅ Released Features
- Multi-provider model testing (7 providers, 18 models)
- Classification task support
- Smart stopping for model families
- Cost comparison and savings calculation
- Sample size control
- JSONL dataset support
- OpenRouter API integration

## Version 0.2.0 (Q2 2024)

### Core Features
- [ ] **Async Evaluation** - Test multiple models in parallel for 5-10x speedup
- [ ] **Task Types**
  - [ ] Extraction tasks (NER, information extraction)
  - [ ] Generation tasks (summarization, translation)
  - [ ] Binary classification (yes/no, true/false)
- [ ] **CSV Support** - Direct CSV file input without conversion
- [ ] **Caching Layer** - Reuse evaluations across runs, save on API costs
- [ ] **Custom Metrics** - Beyond accuracy: F1, precision, recall, BLEU

### Developer Experience
- [ ] **Progress Bars** - Real-time progress for long evaluations
- [ ] **Detailed Logs** - Optional verbose mode for debugging
- [ ] **Retry Configuration** - Customizable retry logic

## Version 0.3.0 (Q3 2024)

### Advanced Features
- [ ] **Prompt Optimization** - Find best prompt + model combination
- [ ] **Multi-step Tasks** - Chain-of-thought optimization
- [ ] **Streaming Support** - For generation tasks
- [ ] **Model Versioning** - Track performance across model updates
- [ ] **Cost Alerts** - Notify when cheaper alternatives become available

### Integration
- [ ] **REST API** - Deploy as a service
- [ ] **CLI Tool** - Command-line interface
- [ ] **GitHub Action** - Automated model selection in CI/CD

## Version 0.4.0 (Q4 2024)

### Platform Features
- [ ] **Web Dashboard** - Visual model comparison
- [ ] **A/B Testing** - Production model comparison framework
- [ ] **RAG Optimization** - Optimize retrieval + generation pipelines
- [ ] **Fine-tuning Advisor** - When to fine-tune vs use larger models

### Ecosystem
- [ ] **Model Marketplace** - Community-contributed model configurations
- [ ] **Benchmark Suite** - Standard benchmarks for comparison
- [ ] **Plugin System** - Custom providers and evaluators

## Version 1.0.0 (2025)

### Enterprise Features
- [ ] **Private Models** - Test your own deployed models
- [ ] **Compliance Checks** - Regulatory requirement validation
- [ ] **SLA Guarantees** - Latency and availability requirements
- [ ] **Budget Management** - Team quotas and spending limits
- [ ] **Audit Logs** - Track all optimization decisions
- [ ] **SSO Integration** - Enterprise authentication

### Scale
- [ ] **Batch Processing** - Handle millions of evaluations
- [ ] **Distributed Testing** - Multi-region evaluation
- [ ] **Model Registry** - Centralized model management

## Long-term Vision

### Research Areas
- **AutoML for LLMs** - Automatic model selection and configuration
- **Cost-aware Fine-tuning** - Optimize fine-tuning vs API costs
- **Multi-objective Optimization** - Balance cost, latency, and quality
- **Federated Evaluation** - Privacy-preserving model testing

### Community Goals
- 100+ supported models
- 10+ provider integrations
- Language-specific SDKs (JavaScript, Go, Rust)
- Academic partnerships for benchmarks

## Contributing

We welcome contributions! Priority areas:
1. New model providers
2. Additional task types
3. Performance optimizations
4. Documentation and examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Feedback

Have ideas or requests? Please open an issue or discussion on GitHub!