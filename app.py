import os
import sys
import json
import yaml
import tempfile
import psutil
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config_parser import BenchmarkConfig, load_config
from src.orchestrator import BenchmarkOrchestrator, load_dataset_prompts
from src.reporter import BenchmarkReporter

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="LLM Performance Benchmarking Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-val {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_hardware_info():
    """Detects system hardware specs for display."""
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Only"
    return {
        "has_cuda": has_cuda,
        "device_name": device_name,
        "ram_gb": total_ram_gb,
        "cpu_count": psutil.cpu_count(logical=True)
    }


def find_saved_result_dirs():
    """Finds existing output directories containing results.csv or results.json."""
    candidates = []
    root = os.path.abspath(os.path.dirname(__file__))
    for item in os.listdir(root):
        full_path = os.path.join(root, item)
        if os.path.isdir(full_path):
            if os.path.exists(os.path.join(full_path, "results.csv")) or os.path.exists(os.path.join(full_path, "results.json")):
                candidates.append(item)
    return candidates or ["benchmark_results"]


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.title("⚙️ Benchmark Settings")

# 1. Preset or Config File Loader
config_source = st.sidebar.radio(
    "Configuration Mode",
    ["Interactive UI", "Load from Config File"],
    index=0
)

# Preset default values
default_models = [
    "sshleifer/tiny-gpt2",
    "hf-internal-testing/tiny-random-gpt2",
    "distilbert/distilgpt2"
]
default_dataset = "datasets/sample_prompts.jsonl"
default_tokens = 35
default_temp = 0.7
default_top_p = 0.9
default_do_sample = True
default_warmup = 1
default_output_dir = "benchmark_results"

if config_source == "Load from Config File":
    uploaded_config = st.sidebar.file_uploader(
        "Upload Config (.yaml / .json)",
        type=["yaml", "yml", "json"]
    )
    preset_choice = st.sidebar.selectbox(
        "Or choose existing config:",
        ["None", "configs/benchmark_config.yaml", "configs/benchmark_config.json"]
    )

    loaded_cfg_path = None
    if uploaded_config is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_config.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_config.getbuffer())
            loaded_cfg_path = tmp.name
    elif preset_choice != "None" and os.path.exists(preset_choice):
        loaded_cfg_path = preset_choice

    if loaded_cfg_path:
        try:
            parsed = load_config(loaded_cfg_path)
            default_models = parsed.models
            default_dataset = parsed.dataset_path
            default_tokens = parsed.max_new_tokens
            default_temp = parsed.temperature
            default_top_p = parsed.top_p
            default_do_sample = parsed.do_sample
            default_warmup = parsed.warmup_runs
            default_output_dir = parsed.output_dir
            st.sidebar.success(f"Loaded {len(default_models)} models from config!")
        except Exception as e:
            st.sidebar.error(f"Config load error: {e}")

# Model Selection
st.sidebar.subheader("1. Target Models (Min. 3)")
preset_model_options = [
    "sshleifer/tiny-gpt2",
    "hf-internal-testing/tiny-random-gpt2",
    "distilbert/distilgpt2",
    "gpt2",
    "facebook/opt-125m",
    "EleutherAI/gpt-neo-125m"
]
selected_models = st.sidebar.multiselect(
    "Choose from standard models:",
    options=preset_model_options,
    default=[m for m in default_models if m in preset_model_options]
)
custom_models_input = st.sidebar.text_input(
    "Add custom Hugging Face model IDs (comma-separated):",
    placeholder="e.g. gpt2-medium, Qwen/Qwen2.5-0.5B"
)
if custom_models_input.strip():
    custom_list = [m.strip() for m in custom_models_input.split(",") if m.strip()]
    for cm in custom_list:
        if cm not in selected_models:
            selected_models.append(cm)

# Prompt Dataset Selection
st.sidebar.subheader("2. Dataset Selection")
dataset_option = st.sidebar.selectbox(
    "Dataset source:",
    ["Sample JSONL (datasets/sample_prompts.jsonl)", "Sample CSV (datasets/sample_prompts.csv)", "Upload Custom Dataset"]
)

dataset_path_to_use = "datasets/sample_prompts.jsonl"
if dataset_option == "Sample JSONL (datasets/sample_prompts.jsonl)":
    dataset_path_to_use = "datasets/sample_prompts.jsonl"
elif dataset_option == "Sample CSV (datasets/sample_prompts.csv)":
    dataset_path_to_use = "datasets/sample_prompts.csv"
else:
    uploaded_data = st.sidebar.file_uploader("Upload dataset (.jsonl or .csv)", type=["jsonl", "csv"])
    if uploaded_data is not None:
        ext = uploaded_data.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_data:
            tmp_data.write(uploaded_data.getbuffer())
            dataset_path_to_use = tmp_data.name
            st.sidebar.info(f"Custom dataset uploaded ({uploaded_data.name})")

# Prompt sample limit (for quick testing)
limit_prompts = st.sidebar.checkbox("Limit prompts count (fast testing)", value=False)
max_prompts_count = 5
if limit_prompts:
    max_prompts_count = st.sidebar.slider("Number of prompts to benchmark", min_value=1, max_value=20, value=3)

# Hyperparameters
st.sidebar.subheader("3. Generation Parameters")
max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=5, max_value=128, value=int(default_tokens), step=5)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=float(default_temp), step=0.05)
top_p = st.sidebar.slider("Top-p (Nucleus)", min_value=0.1, max_value=1.0, value=float(default_top_p), step=0.05)
do_sample = st.sidebar.checkbox("Do Sample", value=bool(default_do_sample))

# Benchmark Run Settings
st.sidebar.subheader("4. Execution & Hardware")
warmup_runs = st.sidebar.slider("Warmup Passes per Model", min_value=0, max_value=3, value=int(default_warmup))
hw = get_hardware_info()
device_choices = ["auto (recommended)", "cpu"]
if hw["has_cuda"]:
    device_choices.append("cuda")
device_selection = st.sidebar.selectbox("Compute Device", device_choices)
device_val = None if device_selection.startswith("auto") else device_selection

output_dir = st.sidebar.text_input("Output Directory", value=default_output_dir)


# ---------------------------------------------------------
# Main Dashboard Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">⚡ LLM Performance Benchmarking Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Profile inference latency, throughput (TPS), peak RAM/VRAM, and lexical diversity across Hugging Face models.</div>', unsafe_allow_html=True)

# Hardware status bar
hw_col1, hw_col2, hw_col3, hw_col4 = st.columns(4)
with hw_col1:
    st.metric("System RAM", f"{hw['ram_gb']} GB")
with hw_col2:
    st.metric("CPU Cores", f"{hw['cpu_count']} Cores")
with hw_col3:
    st.metric("Hardware Acceleration", "CUDA GPU" if hw["has_cuda"] else "CPU Only")
with hw_col4:
    st.metric("Device Name", hw["device_name"][:20])

st.divider()

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab_benchmark, tab_history, tab_dataset, tab_docs = st.tabs([
    "🚀 Run Benchmark",
    "📊 Saved Results & Charts",
    "📝 Dataset Explorer",
    "ℹ️ CLI Generator & Docs"
])

# ---------------------------------------------------------
# Tab 1: Run Benchmark
# ---------------------------------------------------------
with tab_benchmark:
    st.subheader("Benchmark Execution")

    # Contract verification warning
    is_valid_model_count = len(selected_models) >= 3
    if not is_valid_model_count:
        st.warning(f"⚠️ **Requirement:** Benchmark contract requires **at least 3 distinct models**. Currently selected: {len(selected_models)}.")

    # Overview Expander
    with st.expander("📋 Review Active Configuration", expanded=False):
        st.json({
            "models": selected_models,
            "dataset_path": dataset_path_to_use,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "warmup_runs": warmup_runs,
            "device": device_val or "auto-detect",
            "output_dir": output_dir
        })

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        start_benchmark = st.button(
            "▶ Start Benchmark",
            type="primary",
            use_container_width=True,
            disabled=not is_valid_model_count
        )
    with col_hint:
        if not is_valid_model_count:
            st.caption("👈 Please select at least 3 models in the sidebar to enable benchmark execution.")
        else:
            st.caption(f"Ready to benchmark **{len(selected_models)} models** with **{max_new_tokens} tokens/prompt**.")

    # Execution State
    if start_benchmark:
        # Prepare prompts dataset
        prompts = load_dataset_prompts(dataset_path_to_use)
        if limit_prompts and max_prompts_count < len(prompts):
            prompts = prompts[:max_prompts_count]
            # Write trimmed dataset to temporary path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w", encoding="utf-8") as f_tmp:
                for p in prompts:
                    f_tmp.write(json.dumps(p) + "\n")
                effective_dataset_path = f_tmp.name
        else:
            effective_dataset_path = dataset_path_to_use

        config = BenchmarkConfig(
            models=selected_models,
            dataset_path=effective_dataset_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            output_dir=output_dir,
            warmup_runs=warmup_runs,
            device=device_val
        )

        progress_container = st.container()
        with progress_container:
            st.write("---")
            st.markdown("### 🔄 Benchmarking in Progress...")
            overall_progress = st.progress(0.0)
            status_text = st.empty()
            live_metrics_table = st.empty()

            total_steps = len(selected_models) * len(prompts)
            step_counter = {"current": 0}
            live_records = []

            def on_progress(event_data: dict):
                event_type = event_data.get("event")
                if event_type == "model_start":
                    status_text.info(
                        f"**Model [{event_data['model_idx']}/{event_data['total_models']}]:** Loading `{event_data['model_id']}`..."
                    )
                elif event_type == "prompt_done":
                    step_counter["current"] += 1
                    pct = min(1.0, step_counter["current"] / max(total_steps, 1))
                    overall_progress.progress(pct)

                    rec = event_data["record"]
                    live_records.append({
                        "Model": rec["model_name"].split("/")[-1],
                        "Prompt #": rec["prompt_id"],
                        "Latency (s)": f"{rec['latency_sec']:.2f}",
                        "Throughput (tok/s)": f"{rec['throughput_tps']:.1f}",
                        "Peak RAM (MB)": f"{rec['peak_ram_mb']:.1f}",
                        "TTR (Diversity)": f"{rec['type_token_ratio']:.2f}",
                        "Degenerate": "⚠️ Yes" if rec["is_degenerate"] else "✅ No"
                    })
                    status_text.markdown(
                        f"**Evaluating:** `{rec['model_name']}` | Prompt {event_data['prompt_idx']}/{event_data['total_prompts']} | **Latency:** `{rec['latency_sec']:.2f}s` | **TPS:** `{rec['throughput_tps']:.1f}`"
                    )
                    live_metrics_table.dataframe(pd.DataFrame(live_records), use_container_width=True, height=220)
                elif event_type == "model_error":
                    status_text.error(f"❌ Failed to load model `{event_data['model_id']}`: {event_data.get('error')}")

            # Run Orchestrator
            try:
                orchestrator = BenchmarkOrchestrator(config, progress_callback=on_progress)
                df_results = orchestrator.run()
                st.session_state["latest_results"] = df_results
                st.session_state["latest_output_dir"] = output_dir
                overall_progress.progress(1.0)
                status_text.success("🎉 Benchmark run completed successfully!")
            except Exception as ex:
                st.error(f"Benchmark error: {ex}")
                st.stop()

    # Display Results if available
    results_to_show = st.session_state.get("latest_results", None)
    current_out_dir = st.session_state.get("latest_output_dir", output_dir)

    if results_to_show is not None and not results_to_show.empty:
        st.divider()
        st.markdown("### 🏆 Benchmark Summary & Key Performance Indicators")

        reporter = BenchmarkReporter(output_dir=current_out_dir)
        summary_df = reporter.get_summary_dataframe(results_to_show)

        # KPI Metric Cards
        fastest_model = results_to_show.groupby("model_name")["latency_sec"].mean().idxmin()
        fastest_lat = results_to_show.groupby("model_name")["latency_sec"].mean().min()
        highest_tps_model = results_to_show.groupby("model_name")["throughput_tps"].mean().idxmax()
        highest_tps = results_to_show.groupby("model_name")["throughput_tps"].mean().max()
        lowest_ram_model = results_to_show.groupby("model_name")["peak_ram_mb"].max().idxmin()
        lowest_ram = results_to_show.groupby("model_name")["peak_ram_mb"].max().min()
        mean_diversity = results_to_show["type_token_ratio"].mean()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">⚡ Lowest Latency</div>
                <div class="metric-val">{fastest_lat:.2f}s</div>
                <div style="font-size:0.8rem; color:#64748b;">{fastest_model.split('/')[-1]}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🚀 Highest Throughput</div>
                <div class="metric-val">{highest_tps:.1f} TPS</div>
                <div style="font-size:0.8rem; color:#64748b;">{highest_tps_model.split('/')[-1]}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">💾 Lowest Peak RAM</div>
                <div class="metric-val">{lowest_ram:.0f} MB</div>
                <div style="font-size:0.8rem; color:#64748b;">{lowest_ram_model.split('/')[-1]}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">✨ Mean Lexical Diversity</div>
                <div class="metric-val">{mean_diversity:.2f}</div>
                <div style="font-size:0.8rem; color:#64748b;">TTR (0 to 1.0)</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("#### Tabulated Summary Table")
        st.dataframe(summary_df, use_container_width=True)

        # Downloads
        csv_file = os.path.join(current_out_dir, "results.csv")
        json_file = os.path.join(current_out_dir, "results.json")
        dl_col1, dl_col2, _ = st.columns([1, 1, 3])
        if os.path.exists(csv_file):
            with open(csv_file, "rb") as f:
                dl_col1.download_button(
                    "📥 Download results.csv",
                    data=f.read(),
                    file_name="results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        if os.path.exists(json_file):
            with open(json_file, "rb") as f:
                dl_col2.download_button(
                    "📥 Download results.json",
                    data=f.read(),
                    file_name="results.json",
                    mime="application/json",
                    use_container_width=True
                )

        # Render Generated Matplotlib Visualizations
        st.markdown("#### Comparative Visualizations")
        chart_paths = [
            os.path.join(current_out_dir, "latency_comparison.png"),
            os.path.join(current_out_dir, "memory_comparison.png"),
            os.path.join(current_out_dir, "throughput_and_diversity.png")
        ]
        img_tabs = st.tabs(["⏱️ Latency", "💾 Memory (RAM/VRAM)", "⚡ Throughput & Quality"])
        for idx, (tab, pth) in enumerate(zip(img_tabs, chart_paths)):
            with tab:
                if os.path.exists(pth):
                    st.image(pth, use_container_width=True)
                else:
                    st.info(f"Chart `{os.path.basename(pth)}` not found.")

        # Raw Telemetry Table
        with st.expander("🔍 View Raw Prompt-by-Prompt Telemetry"):
            st.dataframe(results_to_show, use_container_width=True)


# ---------------------------------------------------------
# Tab 2: Saved Results & Charts
# ---------------------------------------------------------
with tab_history:
    st.subheader("Saved Benchmark History")
    saved_dirs = find_saved_result_dirs()
    chosen_dir = st.selectbox("Select Benchmark Run Directory:", saved_dirs)

    if chosen_dir and os.path.exists(chosen_dir):
        hist_csv = os.path.join(chosen_dir, "results.csv")
        hist_json = os.path.join(chosen_dir, "results.json")

        if os.path.exists(hist_csv):
            hist_df = pd.read_csv(hist_csv)
            rep = BenchmarkReporter(output_dir=chosen_dir)
            st.markdown(f"**Loaded run records from `{chosen_dir}`** ({len(hist_df)} prompt evaluations across {hist_df['model_name'].nunique()} models)")
            st.dataframe(rep.get_summary_dataframe(hist_df), use_container_width=True)

            hist_tabs = st.tabs(["⏱️ Latency Chart", "💾 Memory Chart", "⚡ Throughput & Diversity", "📋 Raw CSV Data"])
            with hist_tabs[0]:
                lat_img = os.path.join(chosen_dir, "latency_comparison.png")
                if os.path.exists(lat_img):
                    st.image(lat_img, use_container_width=True)
            with hist_tabs[1]:
                mem_img = os.path.join(chosen_dir, "memory_comparison.png")
                if os.path.exists(mem_img):
                    st.image(mem_img, use_container_width=True)
            with hist_tabs[2]:
                tp_img = os.path.join(chosen_dir, "throughput_and_diversity.png")
                if os.path.exists(tp_img):
                    st.image(tp_img, use_container_width=True)
            with hist_tabs[3]:
                st.dataframe(hist_df, use_container_width=True)
        else:
            st.info(f"No results.csv found in directory '{chosen_dir}'.")


# ---------------------------------------------------------
# Tab 3: Dataset Explorer
# ---------------------------------------------------------
with tab_dataset:
    st.subheader("Prompt Dataset Explorer")
    data_files = ["datasets/sample_prompts.jsonl", "datasets/sample_prompts.csv"]
    chosen_file = st.selectbox("Inspect Dataset:", data_files)

    if os.path.exists(chosen_file):
        prompts_list = load_dataset_prompts(chosen_file)
        st.write(f"Total Prompts: **{len(prompts_list)}**")
        df_prompts = pd.DataFrame(prompts_list)
        st.dataframe(df_prompts, use_container_width=True)

        st.markdown("#### Add New Prompt to Dataset")
        with st.form("add_prompt_form"):
            new_prompt_text = st.text_area("Prompt Text:", placeholder="e.g. Compare microservices and monolithic architectures.")
            submit_new = st.form_submit_button("➕ Append Prompt to Dataset")
            if submit_new:
                if new_prompt_text.strip():
                    new_id = len(prompts_list) + 1
                    if chosen_file.endswith(".jsonl"):
                        with open(chosen_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"id": new_id, "prompt": new_prompt_text.strip()}) + "\n")
                    else:
                        with open(chosen_file, "a", encoding="utf-8") as f:
                            f.write(f'\n{new_id},"{new_prompt_text.strip()}"')
                    st.success(f"Prompt #{new_id} added successfully!")
                    st.rerun()
                else:
                    st.warning("Please enter non-empty prompt text.")


# ---------------------------------------------------------
# Tab 4: CLI Generator & Docs
# ---------------------------------------------------------
with tab_docs:
    st.subheader("CLI Command Generator & Quickstart")
    st.markdown("Generate the equivalent CLI command reflecting your current settings from the sidebar:")

    cli_cmd = f"python src/cli.py --config configs/benchmark_config.yaml"
    if max_new_tokens != 35:
        cli_cmd += f" --tokens {max_new_tokens}"
    if warmup_runs != 1:
        cli_cmd += f" --warmup {warmup_runs}"
    if device_val:
        cli_cmd += f" --device {device_val}"
    if output_dir != "benchmark_results":
        cli_cmd += f" --output-dir {output_dir}"

    st.code(cli_cmd, language="bash")

    st.markdown("""
    ### CLI & Docker Commands
    - **Run via Streamlit Web UI**:
      ```bash
      streamlit run app.py
      ```
    - **Run Unit Tests**:
      ```bash
      python -m pytest tests/ -v
      ```
    - **Run Containerized via Docker Compose**:
      ```bash
      docker compose run app python src/cli.py --config configs/benchmark_config.yaml
      ```
    """)

