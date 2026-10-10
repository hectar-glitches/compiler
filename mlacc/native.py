from .cost_model import HardwareConfig
from .isa import Compute, Load, Store

try:
    from ._native_simulator import estimate_cycles as _estimate_cycles
except ImportError as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def estimate_cycles(instructions, hardware_config: HardwareConfig):
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "The native simulator is not built. Run `python setup.py build_ext --inplace`."
        ) from _IMPORT_ERROR

    metadata = []
    for instruction in instructions:
        if isinstance(instruction, (Load, Store)):
            metadata.append({"kind": "load" if isinstance(instruction, Load) else "store",
                             "size_bytes": instruction.size_bytes})
        elif isinstance(instruction, Compute):
            metadata.append({"kind": "compute", "flops": instruction.flops})
        else:
            raise TypeError(f"unsupported instruction: {type(instruction).__name__}")
    cycles = _estimate_cycles(
        metadata,
        hardware_config.compute_flops_per_cycle,
        hardware_config.memory_bandwidth_bytes_per_cycle,
        hardware_config.fixed_memory_latency_cycles,
    )
    return int(cycles) if cycles.is_integer() else cycles
