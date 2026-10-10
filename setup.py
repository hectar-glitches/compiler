from setuptools import Extension, setup
import pybind11


setup(
    name="mlacc",
    ext_modules=[
        Extension(
            "mlacc._native_simulator",
            ["mlacc/native_simulator.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
        )
    ],
)
