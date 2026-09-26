"""
These tests assume the ISA field names from isa.py's hints:
    Load(tensor_name, size_bytes)
    Store(tensor_name, size_bytes)
    Compute(op_type, input_names, output_name, output_size_bytes, flops)
and HardwareConfig's fields from cost_model.py's hints:
    compute_flops_per_cycle, memory_bandwidth_bytes_per_cycle,
    fixed_memory_latency_cycles, on_chip_buffer_bytes

If you named things differently, adjust these tests to match -- but these
exact numbers are hand-calculated against the hinted field names, so
matching them gets you a working self-check for free.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mlacc.cost_model import HardwareConfig
from mlacc.isa import Compute, Load, Store
from mlacc.simulator import Simulator


def make_config():
    return HardwareConfig(
        compute_flops_per_cycle=4.0,
        memory_bandwidth_bytes_per_cycle=2.0,
        fixed_memory_latency_cycles=10.0,
        on_chip_buffer_bytes=1024,
    )


def test_load_then_compute_then_store_correctness():
    # x = [1, 2, 3, 4]; y = relu(x - 10) -> should be all zeros.
    config = make_config()
    initial_memory = {"x": np.array([1.0, 2.0, 3.0, 4.0])}
    sim = Simulator(config, initial_memory)

    instructions = [
        Load(tensor_name="x", size_bytes=16),
        Compute(
            op_type="relu",
            input_names=["x"],
            output_name="y",
            output_size_bytes=16,
            flops=4,
        ),
        Store(tensor_name="y", size_bytes=16),
    ]
    final_memory, cycles = sim.run(instructions)

    assert np.allclose(final_memory["y"], [0.0, 0.0, 0.0, 0.0])
    assert cycles > 0


def test_load_cycle_cost():
    # A single Load of 16 bytes: bandwidth term (16 / 2.0 = 8 cycles)
    # + fixed latency (10 cycles) = 18 cycles, nothing else in the program.
    config = make_config()
    sim = Simulator(config, {"x": np.zeros(4)})
    _, cycles = sim.run([Load(tensor_name="x", size_bytes=16)])
    assert cycles == 18, f"expected 18 cycles, got {cycles}"


def test_fused_vs_unfused_instruction_count_affects_cycles():
    # Same computation (add then relu), but the unfused version has an
    # extra Store+Load round trip in the middle that the fused version
    # skips entirely -- fused MUST take fewer cycles for the same result.
    config = make_config()

    unfused = [
        Load(tensor_name="x", size_bytes=16),
        Load(tensor_name="b", size_bytes=16),
        Compute(op_type="add", input_names=["x", "b"], output_name="h1",
                 output_size_bytes=16, flops=4),
        Store(tensor_name="h1", size_bytes=16),
        Load(tensor_name="h1", size_bytes=16),
        Compute(op_type="relu", input_names=["h1"], output_name="h2",
                 output_size_bytes=16, flops=4),
        Store(tensor_name="h2", size_bytes=16),
    ]
    fused = [
        Load(tensor_name="x", size_bytes=16),
        Load(tensor_name="b", size_bytes=16),
        Compute(op_type="fused", input_names=["x", "b"], output_name="h2",
                 output_size_bytes=16, flops=8, sub_ops=["add", "relu"]),
        Store(tensor_name="h2", size_bytes=16),
    ]

    init = {"x": np.array([1.0, -1.0, 2.0, -2.0]), "b": np.array([0.5] * 4)}
    sim1 = Simulator(config, dict(init))
    final1, unfused_cycles = sim1.run(unfused)

    sim2 = Simulator(config, dict(init))
    final2, fused_cycles = sim2.run(fused)

    assert np.allclose(final1["h2"], final2["h2"]), (
        "fused and unfused pipelines must produce identical output"
    )
    assert fused_cycles < unfused_cycles, (
        f"expected fused ({fused_cycles}) < unfused ({unfused_cycles})"
    )


if __name__ == "__main__":
    import inspect

    tests = [
        (name, fn) for name, fn in list(globals().items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
