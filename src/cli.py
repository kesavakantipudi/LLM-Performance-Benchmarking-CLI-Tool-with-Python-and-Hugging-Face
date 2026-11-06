import argparse
import sys
import os
import logging

# Ensure project root is on sys.path regardless of execution directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_parser import load_config
from src.orchestrator import BenchmarkOrchestrator


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-benchmark",
        description="LLM Performance Benchmarking CLI Tool - Profile Latency, Throughput, Memory, and Quality across Hugging Face models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML (.yaml, .yml) or JSON (.json) configuration file defining models and dataset."
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Override output directory for persistence of results and visualization charts."
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="Override maximum new tokens to generate per prompt."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override number of warmup runs per model prior to metric recording."
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Override device selection ('cuda' or 'cpu'). Default is auto-detect."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging."
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        config = load_config(args.config)

        # Apply CLI overrides if specified
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.tokens:
            config.max_new_tokens = args.tokens
        if args.warmup is not None:
            config.warmup_runs = args.warmup
        if args.device:
            config.device = args.device

        orchestrator = BenchmarkOrchestrator(config)
        orchestrator.run()

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
