#!/bin/bash
# Bullet-proof benchmark script for LLMuxer
# Generates reproducible results with fixed models, dataset, and API endpoints

set -e  # Exit on any error

# Configuration
BENCHMARK_DATE=$(date +%Y%m%d)
BENCHMARK_DIR="benchmarks"
RESULTS_FILE="${BENCHMARK_DIR}/bench_${BENCHMARK_DATE}.json"
MARKDOWN_FILE="docs/benchmarks.md"

echo "🎯 LLMuxer Benchmark Suite"
echo "=========================="
echo "Date: $(date)"
echo "Results file: ${RESULTS_FILE}"
echo "Markdown file: ${MARKDOWN_FILE}"
echo ""

# Check prerequisites
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY environment variable not set"
    echo "   Get your key from: https://openrouter.ai/keys"
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

# Create benchmark directory
mkdir -p "$BENCHMARK_DIR"

echo "📊 Running benchmark with fixed parameters..."
echo "   • Fixed model list (8 models)"
echo "   • Fixed dataset: data/jobs_50.jsonl (50 samples)"
echo "   • Fixed baseline: openai/gpt-4o"
echo "   • Fixed OpenRouter API endpoint"
echo ""

# Run the benchmark
python scripts/bench.py --output "$RESULTS_FILE" --markdown "$MARKDOWN_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Benchmark completed successfully!"
    echo "   Results: ${RESULTS_FILE}"
    echo "   Markdown: ${MARKDOWN_FILE}"
    echo ""
    echo "📈 Summary:"
    python -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
print(f\"   • Models tested: {len(data['results'])}\")
print(f\"   • Baseline model: {data['config']['baseline_model']}\"")
print(f\"   • Dataset size: {data['config']['dataset_size']} samples\"")
print(f\"   • Best model: {data['summary']['best_model']['model']}\"")
print(f\"   • Cost savings: {data['summary']['best_model']['savings_percent']:.1f}%\"")
"
else
    echo "❌ Benchmark failed"
    exit 1
fi