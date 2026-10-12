"""Benchmark materialized attention against PyTorch scaled dot-product attention.

This script targets CUDA environments such as Kaggle or Colab. It records
latency and peak memory for each sequence length in a CSV that can be
downloaded and compared with the local mlacc simulator.
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


def materialized_attention(query, key, value):
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value)


def _run_timed(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        function()
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event))

    return statistics.median(samples), torch.cuda.max_memory_allocated() / (1024**2)


def benchmark_case(batch_size, heads, sequence_length, head_dim, dtype, warmup, repeats):
    shape = (batch_size, heads, sequence_length, head_dim)
    query = torch.randn(shape, device="cuda", dtype=dtype)
    key = torch.randn(shape, device="cuda", dtype=dtype)
    value = torch.randn(shape, device="cuda", dtype=dtype)

    materialized_output = materialized_attention(query, key, value)
    sdpa_output = functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(materialized_output, sdpa_output, rtol=5e-2, atol=5e-2)

    results = []
    for implementation, function in (
        ("materialized", lambda: materialized_attention(query, key, value)),
        ("sdpa", lambda: functional.scaled_dot_product_attention(query, key, value)),
    ):
        try:
            milliseconds, peak_memory_mb = _run_timed(function, warmup, repeats)
            results.append(
                {
                    "implementation": implementation,
                    "status": "ok",
                    "median_ms": f"{milliseconds:.6f}",
                    "peak_memory_mb": f"{peak_memory_mb:.3f}",
                }
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            results.append(
                {
                    "implementation": implementation,
                    "status": "oom",
                    "median_ms": "",
                    "peak_memory_mb": "",
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/flash_attention.csv"))
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required; enable a GPU accelerator in Kaggle or Colab")
    if min(args.sequence_lengths) <= 0 or min(args.batch_size, args.heads, args.head_dim) <= 0:
        parser.error("dimensions and sequence lengths must be positive")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(0)
    rows = []

    for sequence_length in args.sequence_lengths:
        print(f"benchmarking sequence length {sequence_length}")
        try:
            case_results = benchmark_case(
                args.batch_size,
                args.heads,
                sequence_length,
                args.head_dim,
                dtype,
                args.warmup,
                args.repeats,
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            case_results = [
                {
                    "implementation": implementation,
                    "status": "oom",
                    "median_ms": "",
                    "peak_memory_mb": "",
                }
                for implementation in ("materialized", "sdpa")
            ]

        for result in case_results:
            rows.append(
                {
                    "gpu": device_name,
                    "platform": platform.platform(),
                    "pytorch": torch.__version__,
                    "cuda": torch.version.cuda or "",
                    "batch_size": args.batch_size,
                    "heads": args.heads,
                    "sequence_length": sequence_length,
                    "head_dim": args.head_dim,
                    "dtype": args.dtype,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    **result,
                }
            )

    fieldnames = list(rows[0])
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
