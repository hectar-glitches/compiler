import argparse
import time

import numpy as np

try:
    from ._native_simulator import matmul_naive, matmul_neon, matmul_tiled
except ImportError as exc:
    raise ImportError(
        "Build the native extension first: python setup.py build_ext --inplace"
    ) from exc


def _benchmark(function, a, b, repeats):
    result = function(a, b)
    start = time.perf_counter()
    for _ in range(repeats):
        result = function(a, b)
    elapsed = (time.perf_counter() - start) / repeats
    return result, elapsed


def run_benchmark(m, n, k, repeats, tile_size):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    reference, numpy_time = _benchmark(np.matmul, a, b, repeats)
    implementations = [
        ("naive", lambda x, y: matmul_naive(x, y)),
        ("tiled", lambda x, y: matmul_tiled(x, y, tile_size)),
        ("neon", lambda x, y: matmul_neon(x, y)),
    ]

    print(f"matrices: ({m}, {k}) x ({k}, {n}), repeats={repeats}")
    print(f"{'implementation':<16} {'ms':>12} {'vs numpy':>12}")
    print(f"{'numpy':<16} {numpy_time * 1000:>12.3f} {1.0:>12.2f}x")
    for name, function in implementations:
        result, elapsed = _benchmark(function, a, b, repeats)
        np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-4)
        print(
            f"{name:<16} {elapsed * 1000:>12.3f} "
            f"{numpy_time / elapsed:>12.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark the C++ matmul ladder")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.repeats, args.tile_size) <= 0:
        parser.error("all numeric arguments must be positive")
    run_benchmark(args.m, args.n, args.k, args.repeats, args.tile_size)


if __name__ == "__main__":
    main()
