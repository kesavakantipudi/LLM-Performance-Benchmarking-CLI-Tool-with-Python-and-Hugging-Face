# LLM Performance Benchmarking CLI Tool

A production-grade Python Command-Line Interface (CLI) application to systematically benchmark and compare the performance, memory footprint, throughput, and generation quality of Large Language Models (LLMs) from the Hugging Face Hub.

---

## Key Features

- **Standardized CLI Engine (`argparse`)**: Run benchmarks dynamically using YAML or JSON configuration files with CLI overrides for token count, warmup passes, and target devices.
- **Interactive Web Studio (`streamlit`)**: Full-featured graphical dashboard (`app.py`) featuring model pickers, dataset managers, real-time benchmark progress, live telemetry metrics, KPI summary cards, and visualization display.
- **Dynamic Model Loading & Agnostic Execution**: Dynamically downloads, caches, and runs models via Hugging Face `transformers` (`AutoModelForCausalLM` and `AutoTokenizer`) without hardcoding model names.
- **Hardware-Aware Memory & Resource Profiling**: 
  - Tracks peak system RAM via high-frequency background sampling (`psutil`).
  - Automatically detects NVIDIA GPU VRAM (`torch.cuda.max_memory_allocated()`), with seamless CPU fallback.
  - Automatic memory isolation and cleanup (`torch.cuda.empty_cache()` and `gc.collect()`) between model runs to prevent Out-Of-Memory (OOM) accumulation.
- **End-to-End Latency & Throughput**: High-resolution timers (`time.perf_counter()`) isolate generation latency from one-time setup costs and calculate Tokens-Per-Second (TPS).
- **Automated Generation Quality Metrics**: Programmatically evaluates Type-Token Ratio (TTR - Lexical Diversity proxy) and flags empty/degenerate generations.
- **Data Persistence & Visualization**:
  - Persists full raw telemetry to both `results.csv` and `results.json`.
  - Generates publication-ready Matplotlib charts: `latency_comparison.png`, `memory_comparison.png`, and `throughput_and_diversity.png`.
  - Prints a console summary table.
- **Containerized for Reproducibility**: Includes `Dockerfile` and `docker-compose.yml` for isolated cross-platform execution.

---

## Directory Structure

```text
├── configs/
│   ├── benchmark_config.yaml    # YAML benchmark configuration
│   └── benchmark_config.json    # JSON benchmark configuration
├── datasets/
│   ├── sample_prompts.jsonl     # Sample prompts dataset in JSONL format
│   └── sample_prompts.csv       # Sample prompts dataset in CSV format
├── src/
│   ├── __init__.py
│   ├── cli.py                   # Main CLI entry point
│   ├── config_parser.py         # Config validation and schema loader
│   ├── model_runner.py          # Hugging Face model lifecycle and inference
│   ├── orchestrator.py          # Benchmarking coordinator and loop
│   ├── profiler.py              # Hardware monitoring and metrics engine
│   └── reporter.py              # Data persistence, tables, and Matplotlib plotting
├── tests/
│   └── test_benchmarking.py     # Unit test suite
├── app.py                       # Interactive Streamlit Web Studio UI
├── .env.example                 # Environment variable templates
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Multi-container orchestration
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

---

## Installation & Setup

### Local Environment

1. Clone or navigate to the repository directory:
   ```bash
   cd "LLM Performance Benchmarking CLI Tool with Python and Hugging Face"
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional for Hugging Face token):
   ```bash
   cp .env.example .env
   ```

---

## Usage Guide

### 1. Interactive Web Studio (Streamlit UI)

Launch the visual benchmarking studio in your browser:
```bash
streamlit run app.py
```
This opens the web interface where you can:
- Select from preset models or add custom Hugging Face model IDs.
- Upload custom prompt datasets (`.jsonl` or `.csv`) or explore existing ones.
- Adjust generation hyper-parameters (`max_new_tokens`, `temperature`, `top_p`).
- Watch live benchmark progress bars and incremental telemetry table.
- Inspect KPI metric cards (fastest latency, highest TPS, lowest RAM, lexical diversity).
- Browse and download persistence files (`results.csv`, `results.json`) and view Matplotlib charts.

### 2. Basic Execution with CLI Configuration File

Run the benchmark using a YAML configuration:
```bash
python src/cli.py --config configs/benchmark_config.yaml
```

Or using a JSON configuration:
```bash
python src/cli.py --config configs/benchmark_config.json
```

### 2. Command-Line Arguments & Overrides

```text
usage: llm-benchmark [-h] -c CONFIG [-o OUTPUT_DIR] [--tokens TOKENS]
                     [--warmup WARMUP] [--device {cuda,cpu}] [-v]

LLM Performance Benchmarking CLI Tool - Profile Latency, Throughput, Memory,
and Quality across Hugging Face models.

options:
  -h, --help            show this help message and exit
  -c, --config CONFIG   Path to YAML (.yaml, .yml) or JSON (.json) configuration file
  -o, --output-dir OUTPUT_DIR
                        Override output directory for results and charts
  --tokens TOKENS       Override maximum new tokens to generate per prompt
  --warmup WARMUP       Override number of warmup runs per model
  --device {cuda,cpu}   Override device selection ('cuda' or 'cpu')
  -v, --verbose         Enable verbose debug logging
```

#### Example with Overrides:
```bash
python src/cli.py --config configs/benchmark_config.yaml --tokens 50 --warmup 2 --output-dir my_results
```

---

## Configuration Schema

A valid configuration requires at least **three distinct Hugging Face model identifiers** and a path to a prompt dataset (`.jsonl` or `.csv`):

```yaml
# configs/benchmark_config.yaml
models:
  - "sshleifer/tiny-gpt2"
  - "hf-internal-testing/tiny-random-gpt2"
  - "distilbert/distilgpt2"

dataset_path: "datasets/sample_prompts.jsonl"
max_new_tokens: 35
temperature: 0.7
top_p: 0.9
do_sample: true
warmup_runs: 1
output_dir: "benchmark_results"
min_tokens_for_degenerate: 5
```

---

## Running with Docker & Docker Compose

To run the benchmarking suite in an isolated, containerized environment:

1. Build and run using Docker Compose:
   ```bash
   docker compose run app python src/cli.py --config configs/benchmark_config.yaml
   ```

2. Or run directly with Docker:
   ```bash
   docker build -t llm-benchmark .
   docker run --rm -v ${PWD}:/app llm-benchmark --config configs/benchmark_config.yaml
   ```

---

## Telemetry & Output Artifacts

Upon completion of the benchmark run, the tool generates:

1. **`results.csv` & `results.json`**:
   - `model_name`: Model identifier
   - `prompt_id` & `prompt_text`: Input prompt
   - `latency_sec`: End-to-end generation latency in seconds
   - `throughput_tps`: Tokens generated per second (TPS)
   - `peak_ram_bytes` & `peak_ram_mb`: Peak resident set size (system RAM)
   - `peak_gpu_bytes` & `peak_gpu_mb`: Peak CUDA VRAM allocation (if GPU available)
   - `type_token_ratio`: Ratio of unique tokens to total tokens (diversity proxy)
   - `is_degenerate`: Boolean flag for degenerate/empty outputs
   - `output_sample`: Preview of generated text

2. **Generated Visualizations**:
   - `latency_comparison.png`: Bar chart of mean latency per model
   - `memory_comparison.png`: Comparison of peak RAM and GPU VRAM per model
   - `throughput_and_diversity.png`: Dual-panel chart showing throughput (tok/s) and lexical diversity (TTR)

3. **Console Summary Table**:
   - Tabulated summary printed directly to stdout using `tabulate`.

---

## Running Tests

Execute the unit test suite with `pytest`:
```bash
python -m pytest tests/ -v
```
