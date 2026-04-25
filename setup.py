from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "rectirization_py",
        ["rectirization/python_bindings.cpp"],
        include_dirs=["rectirization"],
        cxx_std=14,
    ),
]


setup(
    name="boxfit-rectirization",
    version="0.1.0",
    description="Python bindings for the rectirization polygon-to-rectangle algorithm",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
)
