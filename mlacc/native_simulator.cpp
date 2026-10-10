#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define MLACC_HAS_NEON 1
#endif

namespace py = pybind11;

using Matrix = py::array_t<float, py::array::c_style | py::array::forcecast>;

void validate_inputs(const py::buffer_info& a, const py::buffer_info& b) {
    if (a.ndim != 2 || b.ndim != 2 || a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("matrices must be rank-2 with compatible dimensions");
    }
}

py::array_t<float> make_output(const py::buffer_info& a, const py::buffer_info& b) {
    return py::array_t<float>({a.shape[0], b.shape[1]});
}

py::array_t<float> matmul_naive(const Matrix& a, const Matrix& b) {
    auto a_info = a.request();
    auto b_info = b.request();
    validate_inputs(a_info, b_info);
    auto result = make_output(a_info, b_info);
    auto c_info = result.request();
    const auto* a_data = static_cast<const float*>(a_info.ptr);
    const auto* b_data = static_cast<const float*>(b_info.ptr);
    auto* c_data = static_cast<float*>(c_info.ptr);
    const auto m = a_info.shape[0];
    const auto k = a_info.shape[1];
    const auto n = b_info.shape[1];
    std::fill(c_data, c_data + m * n, 0.0f);

    for (py::ssize_t i = 0; i < m; ++i) {
        for (py::ssize_t k_index = 0; k_index < k; ++k_index) {
            const float value = a_data[i * k + k_index];
            for (py::ssize_t j = 0; j < n; ++j) {
                c_data[i * n + j] += value * b_data[k_index * n + j];
            }
        }
    }
    return result;
}

py::array_t<float> matmul_tiled(const Matrix& a, const Matrix& b, const int tile_size) {
    if (tile_size <= 0) {
        throw std::invalid_argument("tile_size must be positive");
    }
    auto a_info = a.request();
    auto b_info = b.request();
    validate_inputs(a_info, b_info);
    auto result = make_output(a_info, b_info);
    auto c_info = result.request();
    const auto* a_data = static_cast<const float*>(a_info.ptr);
    const auto* b_data = static_cast<const float*>(b_info.ptr);
    auto* c_data = static_cast<float*>(c_info.ptr);
    const auto m = a_info.shape[0];
    const auto k = a_info.shape[1];
    const auto n = b_info.shape[1];
    std::fill(c_data, c_data + m * n, 0.0f);

    for (py::ssize_t i0 = 0; i0 < m; i0 += tile_size) {
        for (py::ssize_t k0 = 0; k0 < k; k0 += tile_size) {
            for (py::ssize_t j0 = 0; j0 < n; j0 += tile_size) {
                const auto i_end = std::min(i0 + tile_size, m);
                const auto k_end = std::min(k0 + tile_size, k);
                const auto j_end = std::min(j0 + tile_size, n);
                for (py::ssize_t i = i0; i < i_end; ++i) {
                    for (py::ssize_t k_index = k0; k_index < k_end; ++k_index) {
                        const float value = a_data[i * k + k_index];
                        for (py::ssize_t j = j0; j < j_end; ++j) {
                            c_data[i * n + j] += value * b_data[k_index * n + j];
                        }
                    }
                }
            }
        }
    }
    return result;
}

py::array_t<float> matmul_neon(const Matrix& a, const Matrix& b) {
    auto a_info = a.request();
    auto b_info = b.request();
    validate_inputs(a_info, b_info);
    auto result = make_output(a_info, b_info);
    auto c_info = result.request();
    const auto* a_data = static_cast<const float*>(a_info.ptr);
    const auto* b_data = static_cast<const float*>(b_info.ptr);
    auto* c_data = static_cast<float*>(c_info.ptr);
    const auto m = a_info.shape[0];
    const auto k = a_info.shape[1];
    const auto n = b_info.shape[1];
    std::fill(c_data, c_data + m * n, 0.0f);

    for (py::ssize_t i = 0; i < m; ++i) {
        for (py::ssize_t k_index = 0; k_index < k; ++k_index) {
            const float value = a_data[i * k + k_index];
            const auto scalar = vdupq_n_f32(value);
            py::ssize_t j = 0;
#if MLACC_HAS_NEON
            for (; j + 4 <= n; j += 4) {
                auto accum = vld1q_f32(c_data + i * n + j);
                auto rhs = vld1q_f32(b_data + k_index * n + j);
                accum = vmlaq_f32(accum, scalar, rhs);
                vst1q_f32(c_data + i * n + j, accum);
            }
#endif
            for (; j < n; ++j) {
                c_data[i * n + j] += value * b_data[k_index * n + j];
            }
        }
    }
    return result;
}

double estimate_cycles(
    const py::iterable& instructions,
    const double compute_flops_per_cycle,
    const double memory_bandwidth_bytes_per_cycle,
    const double fixed_memory_latency_cycles) {
    if (compute_flops_per_cycle <= 0.0) {
        throw std::invalid_argument("compute_flops_per_cycle must be positive");
    }
    if (memory_bandwidth_bytes_per_cycle <= 0.0) {
        throw std::invalid_argument(
            "memory_bandwidth_bytes_per_cycle must be positive");
    }
    if (fixed_memory_latency_cycles < 0.0) {
        throw std::invalid_argument(
            "fixed_memory_latency_cycles must be non-negative");
    }

    double cycles = 0.0;
    for (const py::handle item : instructions) {
        const py::dict instruction = py::reinterpret_borrow<py::dict>(item);
        const std::string kind = instruction["kind"].cast<std::string>();
        if (kind == "load" || kind == "store") {
            const auto size_bytes = instruction["size_bytes"].cast<double>();
            cycles += size_bytes / memory_bandwidth_bytes_per_cycle
                + fixed_memory_latency_cycles;
        } else if (kind == "compute") {
            cycles += instruction["flops"].cast<double>()
                / compute_flops_per_cycle;
        } else {
            throw std::invalid_argument("unsupported instruction kind: " + kind);
        }
    }
    return cycles;
}

PYBIND11_MODULE(_native_simulator, module) {
    module.doc() = "Native cycle estimator for the mlacc ISA";
    module.def(
        "estimate_cycles",
        &estimate_cycles,
        py::arg("instructions"),
        py::arg("compute_flops_per_cycle"),
        py::arg("memory_bandwidth_bytes_per_cycle"),
        py::arg("fixed_memory_latency_cycles"));
    module.def("matmul_naive", &matmul_naive, py::arg("a"), py::arg("b"));
    module.def(
        "matmul_tiled",
        &matmul_tiled,
        py::arg("a"),
        py::arg("b"),
        py::arg("tile_size") = 32);
    module.def("matmul_neon", &matmul_neon, py::arg("a"), py::arg("b"));
}
