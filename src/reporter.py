import os
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


class BenchmarkReporter:
    """
    Handles persisting raw telemetry data to CSV/JSON, calculating aggregated
    summary statistics per model, displaying formatted console tables, and
    rendering informative matplotlib visualization charts.
    """

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def persist_results(self, records: List[Dict[str, Any]], format_type: str = "csv") -> str:
        """
        Saves full raw benchmarking telemetry to results.csv or results.json.
        """
        df = pd.DataFrame(records)
        output_file = os.path.join(self.output_dir, f"results.{format_type.lower()}")

        if format_type.lower() == "json":
            df.to_json(output_file, orient="records", indent=2)
        else:
            df.to_csv(output_file, index=False)

        return output_file

    def get_summary_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mean latency, mean throughput, peak RAM, peak GPU VRAM,
        average output length, mean Type-Token Ratio, and degenerate output rate per model.
        """
        has_gpu = "peak_gpu_mb" in df.columns and df["peak_gpu_mb"].notna().any()

        agg_dict = {
            "latency_sec": ["mean", "std"],
            "throughput_tps": ["mean", "std"],
            "peak_ram_mb": "max",
            "generated_tokens": "mean",
            "type_token_ratio": "mean",
            "is_degenerate": lambda x: (x == True).mean() * 100
        }

        if has_gpu:
            agg_dict["peak_gpu_mb"] = "max"

        summary = df.groupby("model_name").agg(agg_dict).reset_index()

        # Flatten multi-index columns
        flat_cols = []
        for col in summary.columns:
            if isinstance(col, tuple):
                if col[1] == "":
                    flat_cols.append(col[0])
                elif "lambda" in col[1]:
                    flat_cols.append("Degenerate Rate (%)")
                else:
                    flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col)
        summary.columns = flat_cols

        # Rename for presentation clarity
        rename_map = {
            "model_name": "Model",
            "latency_sec_mean": "Avg Latency (s)",
            "latency_sec_std": "Latency Std (s)",
            "throughput_tps_mean": "Avg Throughput (tok/s)",
            "throughput_tps_std": "Throughput Std",
            "peak_ram_mb_max": "Peak RAM (MB)",
            "peak_gpu_mb_max": "Peak VRAM (MB)",
            "generated_tokens_mean": "Avg Output Tokens",
            "type_token_ratio_mean": "Avg TTR (Diversity)",
            "is_degenerate_rate_pct": "Degenerate Rate (%)"
        }
        summary = summary.rename(columns={k: v for k, v in rename_map.items() if k in summary.columns})

        # Round numeric values for clean readability
        numeric_cols = summary.select_dtypes(include=["float", "float64", "float32"]).columns
        summary[numeric_cols] = summary[numeric_cols].round(2)

        return summary

    def print_console_summary(self, df: pd.DataFrame) -> None:
        """
        Prints an aesthetically formatted summary table directly to console standard output.
        """
        if df.empty:
            print("\nNo benchmark data collected.")
            return

        summary = self.get_summary_dataframe(df)

        print("\n" + "=" * 95)
        print(" " * 30 + "LLM BENCHMARK PERFORMANCE SUMMARY")
        print("=" * 95)
        print(tabulate(summary, headers="keys", tablefmt="grid", showindex=False))
        print("=" * 95 + "\n")

    def generate_visualizations(self, df: pd.DataFrame) -> List[str]:
        """
        Generates and saves visual comparison charts across tested models:
        1. Average Latency Comparison (latency_comparison.png)
        2. Peak Memory Usage Comparison (memory_comparison.png)
        3. Throughput & Quality Metrics Overview (throughput_and_diversity.png)

        Returns:
            List[str]: Paths to saved image files
        """
        if df.empty:
            return []

        saved_files = []
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Average Latency Comparison
        plt.figure(figsize=(9, 5))
        latency_series = df.groupby("model_name")["latency_sec"].mean()
        ax1 = latency_series.plot(kind="bar", color="#3b82f6", edgecolor="#1d4ed8", width=0.55)
        plt.title("LLM Inference Latency Comparison (Lower is Faster)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Model Name", fontsize=11, labelpad=8)
        plt.ylabel("Mean Latency (Seconds)", fontsize=11, labelpad=8)
        plt.xticks(rotation=15, ha="right", fontsize=10)
        plt.tight_layout()

        # Add data labels
        for p in ax1.patches:
            ax1.annotate(
                f"{p.get_height():.2f}s",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                xytext=(0, 3),
                textcoords="offset points"
            )

        latency_chart_path = os.path.join(self.output_dir, "latency_comparison.png")
        plt.savefig(latency_chart_path, dpi=200)
        plt.close()
        saved_files.append(latency_chart_path)

        # 2. Peak Memory Usage Comparison (RAM and optional VRAM)
        plt.figure(figsize=(9, 5))
        has_gpu = "peak_gpu_mb" in df.columns and df["peak_gpu_mb"].notna().any()

        mem_data = {"Peak RAM (MB)": df.groupby("model_name")["peak_ram_mb"].max()}
        if has_gpu:
            mem_data["Peak GPU VRAM (MB)"] = df.groupby("model_name")["peak_gpu_mb"].max().fillna(0)

        mem_df = pd.DataFrame(mem_data)
        colors = ["#10b981", "#8b5cf6"] if has_gpu else ["#10b981"]
        ax2 = mem_df.plot(kind="bar", color=colors, edgecolor="#047857", width=0.55 if not has_gpu else 0.7)
        plt.title("Peak Memory Allocation per Model (RAM & VRAM)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Model Name", fontsize=11, labelpad=8)
        plt.ylabel("Memory (MB)", fontsize=11, labelpad=8)
        plt.xticks(rotation=15, ha="right", fontsize=10)
        plt.tight_layout()

        for p in ax2.patches:
            val = p.get_height()
            if val > 0:
                ax2.annotate(
                    f"{val:.0f} MB",
                    (p.get_x() + p.get_width() / 2.0, val),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="semibold",
                    xytext=(0, 3),
                    textcoords="offset points"
                )

        memory_chart_path = os.path.join(self.output_dir, "memory_comparison.png")
        plt.savefig(memory_chart_path, dpi=200)
        plt.close()
        saved_files.append(memory_chart_path)

        # 3. Throughput & Quality Comparison
        fig, (ax_tp, ax_div) = plt.subplots(1, 2, figsize=(13, 5))
        tp_series = df.groupby("model_name")["throughput_tps"].mean()
        tp_series.plot(kind="bar", ax=ax_tp, color="#f59e0b", edgecolor="#d97706", width=0.5)
        ax_tp.set_title("Mean Throughput (Tokens / Sec)", fontsize=11, fontweight="bold")
        ax_tp.set_ylabel("Tokens / Second")
        ax_tp.tick_params(axis="x", rotation=15)
        for p in ax_tp.patches:
            ax_tp.annotate(
                f"{p.get_height():.1f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points"
            )

        div_series = df.groupby("model_name")["type_token_ratio"].mean()
        div_series.plot(kind="bar", ax=ax_div, color="#ec4899", edgecolor="#be185d", width=0.5)
        ax_div.set_title("Lexical Diversity (Type-Token Ratio)", fontsize=11, fontweight="bold")
        ax_div.set_ylabel("TTR (0.0 to 1.0)")
        ax_div.set_ylim(0, 1.1)
        ax_div.tick_params(axis="x", rotation=15)
        for p in ax_div.patches:
            ax_div.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points"
            )

        plt.suptitle("Throughput & Output Quality Metrics", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        metrics_chart_path = os.path.join(self.output_dir, "throughput_and_diversity.png")
        plt.savefig(metrics_chart_path, dpi=200, bbox_inches="tight")
        plt.close()
        saved_files.append(metrics_chart_path)

        return saved_files
