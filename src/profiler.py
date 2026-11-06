import os
import re
import time
import threading
from typing import Callable, Tuple, Dict, Any, Optional
import psutil
import torch


def compute_type_token_ratio(text: str) -> float:
    """
    Computes the Type-Token Ratio (unique tokens / total tokens) as a lexical diversity proxy.
    Returns 0.0 if text has no tokens.
    """
    if not text:
        return 0.0
    # Clean and split into lowercased alphanumeric words
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def is_degenerate_or_empty(text: str, token_count: int, min_tokens: int = 5) -> bool:
    """
    Determines if an output is degenerate (empty, whitespace, or shorter than threshold).
    """
    if not text or not text.strip():
        return True
    return token_count < min_tokens


class HardwareSampler:
    """
    Background sampling thread to continuously poll system RAM and GPU VRAM
    during model generation to capture true peak utilization spikes.
    """

    def __init__(self, interval_sec: float = 0.01):
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.peak_ram_bytes: int = 0
        self.peak_gpu_bytes: int = 0
        self.has_gpu = torch.cuda.is_available()

    def _sample_loop(self):
        while not self.stop_event.is_set():
            try:
                # Poll RAM
                ram = self.process.memory_info().rss
                if ram > self.peak_ram_bytes:
                    self.peak_ram_bytes = ram

                # Poll GPU if present
                if self.has_gpu:
                    gpu = torch.cuda.memory_allocated()
                    if gpu > self.peak_gpu_bytes:
                        self.peak_gpu_bytes = gpu
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self):
        # Initialize with baseline
        self.peak_ram_bytes = self.process.memory_info().rss
        if self.has_gpu:
            torch.cuda.reset_peak_memory_stats()
            self.peak_gpu_bytes = torch.cuda.memory_allocated()

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Tuple[int, int]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=0.5)

        # Final check
        try:
            final_ram = self.process.memory_info().rss
            if final_ram > self.peak_ram_bytes:
                self.peak_ram_bytes = final_ram

            if self.has_gpu:
                cuda_peak = torch.cuda.max_memory_allocated()
                if cuda_peak > self.peak_gpu_bytes:
                    self.peak_gpu_bytes = cuda_peak
        except Exception:
            pass

        return self.peak_ram_bytes, self.peak_gpu_bytes


def profile_inference(
    inference_func: Callable[[], Tuple[str, int]],
    min_tokens_for_degenerate: int = 5
) -> Dict[str, Any]:
    """
    Profiles an inference invocation, measuring high-resolution end-to-end latency,
    throughput (Tokens Per Second), peak system RAM (bytes and MB), peak GPU VRAM
    (if GPU available), and quality metrics (Type-Token Ratio and degenerate check).

    Args:
        inference_func: Callable executing generation and returning (output_text, token_count)
        min_tokens_for_degenerate: Minimum acceptable tokens threshold

    Returns:
        Dict[str, Any]: Detailed metrics dictionary
    """
    sampler = HardwareSampler(interval_sec=0.01)
    sampler.start()

    # High resolution latency measurement around generation only
    start_time = time.perf_counter()
    output_text, token_count = inference_func()
    end_time = time.perf_counter()

    peak_ram_bytes, peak_gpu_bytes = sampler.stop()

    latency_sec = max(end_time - start_time, 1e-6)
    throughput_tps = token_count / latency_sec if latency_sec > 0 else 0.0

    # Quality Metrics
    ttr = compute_type_token_ratio(output_text)
    degenerate = is_degenerate_or_empty(output_text, token_count, min_tokens_for_degenerate)

    # Conversion to Megabytes
    peak_ram_mb = round(peak_ram_bytes / (1024 * 1024), 2)
    peak_gpu_mb = round(peak_gpu_bytes / (1024 * 1024), 2) if torch.cuda.is_available() else None

    return {
        "output_text": output_text,
        "generated_tokens": token_count,
        "latency_sec": latency_sec,
        "throughput_tps": round(throughput_tps, 2),
        "peak_ram_bytes": peak_ram_bytes,
        "peak_ram_mb": peak_ram_mb,
        "peak_gpu_bytes": peak_gpu_bytes if torch.cuda.is_available() else None,
        "peak_gpu_mb": peak_gpu_mb,
        "type_token_ratio": ttr,
        "is_degenerate": degenerate
    }
