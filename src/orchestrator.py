import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
from tqdm import tqdm

from src.config_parser import BenchmarkConfig
from src.model_runner import HuggingFaceRunner
from src.profiler import profile_inference
from src.reporter import BenchmarkReporter

logger = logging.getLogger(__name__)


def load_dataset_prompts(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Loads prompts from either a .jsonl or .csv file.
    Returns a list of dicts with at least 'id' and 'prompt' keys.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    _, ext = os.path.splitext(dataset_path)
    ext = ext.lower()
    items = []

    if ext == ".jsonl":
        with open(dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prompt_text = data.get("prompt") or data.get("text") or data.get("instruction")
                prompt_id = data.get("id", idx)
                items.append({"id": prompt_id, "prompt": prompt_text})
    elif ext == ".csv":
        df = pd.read_csv(dataset_path)
        prompt_col = None
        for candidate in ["prompt", "text", "instruction"]:
            if candidate in df.columns:
                prompt_col = candidate
                break
        if not prompt_col:
            raise ValueError(f"Dataset CSV must contain a 'prompt' or 'text' column. Found: {list(df.columns)}")

        id_col = "id" if "id" in df.columns else None
        for idx, row in df.iterrows():
            prompt_id = row[id_col] if id_col else (idx + 1)
            items.append({"id": prompt_id, "prompt": str(row[prompt_col])})
    else:
        raise ValueError(f"Unsupported dataset format '{ext}'. Must be .jsonl or .csv.")

    if not items:
        raise ValueError(f"Dataset at '{dataset_path}' contains zero valid prompt items.")

    return items


class BenchmarkOrchestrator:
    """
    Coordinates the end-to-end benchmark execution:
    - Loads prompts
    - Iterates sequentially across target models with error isolation
    - Executes warmup runs
    - Profiles prompts with real-time tqdm progress feedback
    - Manages memory disposal between models
    - Persists telemetry and generates comparison charts and console summary
    """

    def __init__(self, config: BenchmarkConfig, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.config = config
        self.reporter = BenchmarkReporter(output_dir=config.output_dir)
        self.progress_callback = progress_callback

    def run(self) -> pd.DataFrame:
        logger.info(f"Loading prompts from dataset: {self.config.dataset_path}")
        prompts = load_dataset_prompts(self.config.dataset_path)
        logger.info(f"Successfully loaded {len(prompts)} prompts.")

        all_records: List[Dict[str, Any]] = []

        print(f"\n=======================================================")
        print(f"  Starting LLM Benchmark Suite across {len(self.config.models)} models")
        print(f"  Dataset: {self.config.dataset_path} ({len(prompts)} prompts)")
        print(f"  Max New Tokens: {self.config.max_new_tokens} | Warmup: {self.config.warmup_runs}")
        print(f"=======================================================\n")

        for model_idx, model_id in enumerate(self.config.models, start=1):
            print(f"\n[{model_idx}/{len(self.config.models)}] Initializing model: {model_id}")
            if self.progress_callback:
                self.progress_callback({
                    "event": "model_start",
                    "model_idx": model_idx,
                    "total_models": len(self.config.models),
                    "model_id": model_id
                })

            runner = None
            try:
                runner = HuggingFaceRunner(model_id=model_id, device=self.config.device)
            except Exception as e:
                logger.error(f"Failed to load model '{model_id}': {e}", exc_info=True)
                print(f"  [ERROR] Skipping '{model_id}' due to loading error: {e}")
                if self.progress_callback:
                    self.progress_callback({
                        "event": "model_error",
                        "model_idx": model_idx,
                        "total_models": len(self.config.models),
                        "model_id": model_id,
                        "error": str(e)
                    })
                if runner:
                    runner.unload()
                continue

            # Warmup runs (not included in metrics)
            if self.config.warmup_runs > 0 and len(prompts) > 0:
                warmup_prompt = prompts[0]["prompt"]
                logger.info(f"Executing {self.config.warmup_runs} warmup run(s) for '{model_id}'...")
                for _ in range(self.config.warmup_runs):
                    try:
                        runner.generate(
                            warmup_prompt,
                            max_new_tokens=min(10, self.config.max_new_tokens),
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample
                        )
                    except Exception as e:
                        logger.warning(f"Warmup pass encountered an issue: {e}")

            # Process prompts with tqdm progress bar
            prompt_bar = tqdm(
                prompts,
                desc=f"Benchmarking {model_id.split('/')[-1]}",
                unit="prompt",
                leave=True
            )

            for prompt_idx, prompt_item in enumerate(prompt_bar, start=1):
                p_id = prompt_item["id"]
                p_text = prompt_item["prompt"]

                try:
                    def _inference():
                        return runner.generate(
                            prompt=p_text,
                            max_new_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample
                        )

                    metrics = profile_inference(
                        _inference,
                        min_tokens_for_degenerate=self.config.min_tokens_for_degenerate
                    )

                    record = {
                        "model_name": model_id,
                        "prompt_id": p_id,
                        "prompt_text": p_text[:100] + ("..." if len(p_text) > 100 else ""),
                        "latency_sec": metrics["latency_sec"],
                        "throughput_tps": metrics["throughput_tps"],
                        "generated_tokens": metrics["generated_tokens"],
                        "peak_ram_bytes": metrics["peak_ram_bytes"],
                        "peak_ram_mb": metrics["peak_ram_mb"],
                        "peak_gpu_bytes": metrics["peak_gpu_bytes"],
                        "peak_gpu_mb": metrics["peak_gpu_mb"],
                        "type_token_ratio": metrics["type_token_ratio"],
                        "is_degenerate": metrics["is_degenerate"],
                        "output_sample": metrics["output_text"][:120] + ("..." if len(metrics["output_text"]) > 120 else "")
                    }
                    all_records.append(record)

                    if self.progress_callback:
                        self.progress_callback({
                            "event": "prompt_done",
                            "model_idx": model_idx,
                            "total_models": len(self.config.models),
                            "model_id": model_id,
                            "prompt_id": p_id,
                            "prompt_idx": prompt_idx,
                            "total_prompts": len(prompts),
                            "record": record,
                            "metrics": metrics
                        })

                    prompt_bar.set_postfix({
                        "Lat": f"{metrics['latency_sec']:.2f}s",
                        "TPS": f"{metrics['throughput_tps']:.1f}",
                        "RAM": f"{metrics['peak_ram_mb']}MB"
                    })

                except Exception as ex:
                    logger.error(f"Inference failed on model '{model_id}', prompt {p_id}: {ex}")
                    print(f"\n  [WARN] Inference error on prompt {p_id}: {ex}")

            # Critical: Cleanup model and flush GPU / RAM memory
            runner.unload()

        df_results = pd.DataFrame(all_records)

        if not df_results.empty:
            # Persist results
            csv_path = self.reporter.persist_results(all_records, format_type="csv")
            json_path = self.reporter.persist_results(all_records, format_type="json")
            print(f"\n[Artifact] Raw benchmark telemetry saved to:")
            print(f"  - CSV:  {csv_path}")
            print(f"  - JSON: {json_path}")

            # Generate charts
            charts = self.reporter.generate_visualizations(df_results)
            print(f"[Artifact] Visualizations generated:")
            for chart in charts:
                print(f"  - {chart}")

            # Print console summary
            self.reporter.print_console_summary(df_results)
        else:
            print("\n[WARN] No results gathered from benchmark run.")

        if self.progress_callback:
            self.progress_callback({
                "event": "complete",
                "records": all_records,
                "df": df_results
            })

        return df_results
