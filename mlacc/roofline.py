import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .cost_model import HardwareConfig

STYLE = {
    "add": "#2563eb",
    "relu": "#16a34a",
    "matmul": "#d97706",
    "naive": "#64748b",
    "tiled": "#7c3aed",
    "neon": "#dc2626",
}


def operational_intensity(flops, bytes_moved):
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def roofline_points(config, clock_ghz, max_intensity=1e3):
    peak_flops = config.compute_flops_per_cycle * clock_ghz
    peak_bandwidth = config.memory_bandwidth_bytes_per_cycle * clock_ghz
    intensity = np.logspace(-2, np.log10(max_intensity), 300)
    attainable = np.minimum(peak_flops, intensity * peak_bandwidth)
    return intensity, attainable, peak_flops, peak_bandwidth


def workload_points(m, n, k):
    return {
        "add": (
            operational_intensity(m * n, 3 * m * n * 4),
            m * n,
        ),
        "relu": (
            operational_intensity(m * n, 2 * m * n * 4),
            m * n,
        ),
        "matmul": (
            operational_intensity(2 * m * n * k, 4 * (m * k + k * n + m * n)),
            2 * m * n * k,
        ),
    }


def measured_matmul_points(a, b, repeats):
    try:
        from ._native_simulator import matmul_naive, matmul_neon, matmul_tiled
    except ImportError:
        return {}

    reference = a @ b
    implementations = {
        "naive": matmul_naive,
        "tiled": lambda x, y: matmul_tiled(x, y, 32),
        "neon": matmul_neon,
    }
    points = {}
    for name, function in implementations.items():
        result = function(a, b)
        np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-4)
        start = time.perf_counter()
        for _ in range(repeats):
            function(a, b)
        seconds = (time.perf_counter() - start) / repeats
        flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
        points[name] = {
            "intensity": flops / (4 * (a.size + b.size + result.size)),
            "gflops": flops / seconds / 1e9,
            "seconds": seconds,
        }
    return points


def create_roofline_plot(output, m, n, k, clock_ghz, repeats):
    config = HardwareConfig()
    intensity, attainable, peak_flops, peak_bandwidth = roofline_points(
        config, clock_ghz
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    figure.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#ffffff")
    axis.loglog(
        intensity,
        attainable,
        color="#0f172a",
        label="Modeled roofline",
        linewidth=2.8,
        zorder=2,
    )
    axis.axhline(
        peak_flops,
        color="#be123c",
        linestyle="--",
        linewidth=1.4,
        label=f"Compute ceiling · {peak_flops:.1f} GFLOP/s",
    )
    axis.axvline(
        peak_flops / peak_bandwidth,
        color="#b45309",
        linestyle=":",
        linewidth=1.6,
        label=f"Ridge point · {peak_flops / peak_bandwidth:.1f} FLOP/byte",
    )

    for name, (x_value, _) in workload_points(m, n, k).items():
        y_value = min(peak_flops, x_value * peak_bandwidth)
        axis.scatter(
            x_value,
            y_value,
            s=90,
            color=STYLE[name],
            edgecolor="white",
            linewidth=1.2,
            label=f"{name} model",
            zorder=4,
        )

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    measured = measured_matmul_points(a, b, repeats)
    for name, point in measured.items():
        axis.scatter(
            point["intensity"],
            point["gflops"],
            marker="X",
            s=95,
            color=STYLE[name],
            edgecolor="white",
            linewidth=0.8,
            label=f"{name} measured",
            zorder=5,
        )
        axis.annotate(
            f"{point['gflops']:.1f}",
            (point["intensity"], point["gflops"]),
            xytext=(7, -12),
            textcoords="offset points",
            fontsize=9,
            color=STYLE[name],
        )

    axis.set_title(
        "mlacc performance roofline",
        loc="left",
        fontsize=18,
        weight="bold",
        pad=22,
    )
    axis.text(
        0,
        1.01,
        f"Modeled accelerator · {m}×{n}×{k} matmul measurements",
        transform=axis.transAxes,
        fontsize=10,
        color="#475569",
    )
    axis.set_xlabel("Arithmetic intensity (FLOP/byte)", labelpad=8)
    axis.set_ylabel("Performance (GFLOP/s)", labelpad=8)
    axis.grid(True, which="major", color="#cbd5e1", alpha=0.55)
    axis.grid(True, which="minor", color="#e2e8f0", alpha=0.35)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize=9,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_benchmark_plot(output, m, n, k, repeats):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    measured = measured_matmul_points(a, b, repeats)
    if not measured:
        raise ImportError("build the native extension before plotting benchmarks")

    labels = ["naive", "tiled", "neon"]
    times = [measured[name]["seconds"] * 1000 for name in labels]
    gflops = [measured[name]["gflops"] for name in labels]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (time_axis, performance_axis) = plt.subplots(
        1, 2, figsize=(10, 4.8), constrained_layout=True
    )
    figure.patch.set_facecolor("#f8fafc")
    colors = [STYLE[name] for name in labels]
    x = np.arange(len(labels))

    time_axis.bar(x, times, color=colors, width=0.62)
    time_axis.set_title("Execution time", loc="left", weight="bold")
    time_axis.set_ylabel("Milliseconds")
    time_axis.set_xticks(x, labels)
    time_axis.spines[["top", "right"]].set_visible(False)
    time_axis.grid(axis="y", alpha=0.35)
    for index, value in enumerate(times):
        time_axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    performance_axis.bar(x, gflops, color=colors, width=0.62)
    performance_axis.set_title("Achieved throughput", loc="left", weight="bold")
    performance_axis.set_ylabel("GFLOP/s")
    performance_axis.set_xticks(x, labels)
    performance_axis.spines[["top", "right"]].set_visible(False)
    performance_axis.grid(axis="y", alpha=0.35)
    for index, value in enumerate(gflops):
        performance_axis.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    figure.suptitle(
        f"C++ matmul ladder · {m}×{n}×{k} · {repeats} repeats",
        fontsize=17,
        weight="bold",
        x=0.03,
        ha="left",
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Generate the mlacc roofline plot")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--clock-ghz", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("roofline.png"))
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("matmul_benchmark.png"),
    )
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.repeats, args.clock_ghz) <= 0:
        parser.error("all numeric arguments must be positive")
    create_roofline_plot(
        args.output,
        args.m,
        args.n,
        args.k,
        args.clock_ghz,
        args.repeats,
    )
    create_benchmark_plot(
        args.benchmark_output,
        args.m,
        args.n,
        args.k,
        args.repeats,
    )
    print(f"wrote {args.output} and {args.benchmark_output}")


if __name__ == "__main__":
    main()
