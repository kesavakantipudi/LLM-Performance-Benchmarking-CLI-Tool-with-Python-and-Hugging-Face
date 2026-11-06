import os
import json
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Validated configuration for LLM benchmarking run."""
    models: List[str]
    dataset_path: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    output_dir: str = "benchmark_results"
    warmup_runs: int = 1
    device: Optional[str] = None  # None for auto-detect ('cuda' or 'cpu')
    min_tokens_for_degenerate: int = 5

    def validate(self) -> None:
        """Validate that the configuration satisfies all system contracts."""
        if not self.models or not isinstance(self.models, list):
            raise ValueError("Configuration 'models' must be a non-empty list of model identifiers.")
        
        if len(self.models) < 3:
            raise ValueError(
                f"Configuration must contain at least three distinct model identifiers. Found {len(self.models)}."
            )
        
        if len(set(self.models)) != len(self.models):
            raise ValueError(
                f"Duplicate model identifiers detected in 'models' list: {self.models}"
            )
            
        for model in self.models:
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"Invalid model identifier: '{model}'. Must be a non-empty string.")

        if not self.dataset_path or not isinstance(self.dataset_path, str):
            raise ValueError("Configuration 'dataset_path' must be a valid non-empty string.")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file does not exist at path: '{self.dataset_path}'")

        if self.max_new_tokens <= 0:
            raise ValueError(f"'max_new_tokens' must be a positive integer, got {self.max_new_tokens}.")

        if self.warmup_runs < 0:
            raise ValueError(f"'warmup_runs' must be non-negative, got {self.warmup_runs}.")


def load_config(file_path: str) -> BenchmarkConfig:
    """
    Parses a YAML or JSON configuration file and returns a validated BenchmarkConfig instance.

    Args:
        file_path: Path to YAML (.yaml, .yml) or JSON (.json) configuration file.

    Returns:
        BenchmarkConfig: Validated dataclass instance.

    Raises:
        FileNotFoundError: If configuration file is not found.
        ValueError: If file is unsupported format, malformed, or missing required keys.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext not in [".yaml", ".yml", ".json"]:
        raise ValueError(
            f"Unsupported configuration file extension '{ext}'. Only .yaml, .yml, or .json are supported."
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if ext in [".yaml", ".yml"]:
                raw_data = yaml.safe_load(f)
            else:
                raw_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error parsing configuration file '{file_path}': {e}") from e

    if not isinstance(raw_data, dict):
        raise ValueError(f"Configuration file root must be a dictionary/mapping, got {type(raw_data).__name__}.")

    # Required keys verification
    required_keys = ["models", "dataset_path"]
    missing = [k for k in required_keys if k not in raw_data]
    if missing:
        raise ValueError(f"Missing required configuration key(s): {', '.join(missing)}")

    # Instantiate config with defaults for optional parameters
    config = BenchmarkConfig(
        models=raw_data.get("models"),
        dataset_path=raw_data.get("dataset_path"),
        max_new_tokens=int(raw_data.get("max_new_tokens", 50)),
        temperature=float(raw_data.get("temperature", 0.7)),
        top_p=float(raw_data.get("top_p", 0.9)),
        do_sample=bool(raw_data.get("do_sample", True)),
        output_dir=str(raw_data.get("output_dir", "benchmark_results")),
        warmup_runs=int(raw_data.get("warmup_runs", 1)),
        device=raw_data.get("device"),
        min_tokens_for_degenerate=int(raw_data.get("min_tokens_for_degenerate", 5))
    )

    config.validate()
    return config
