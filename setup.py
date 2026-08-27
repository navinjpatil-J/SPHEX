"""Setuptools configuration for the script-based SPHEX v1.2.0 release."""

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="SPHEX",
    version="1.2.0",
    author="Navinkumar Patil",
    author_email="navinjpatil@gmail.com",
    description=(
        "SPHEX: Spectral Pattern Heterogeneity indeX Analyzer — "
        "a multiscale framework for quantitative AFM biofilm "
        "surface heterogeneity analysis"
    ),
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/navinjpatil-J/SPHEX",
    project_urls={
        "Source": "https://github.com/navinjpatil-J/SPHEX",
        "Issues": "https://github.com/navinjpatil-J/SPHEX/issues",
    },
    license="MIT",
    license_files=["LICENSE"],

    # The repository currently has a flat module layout, not a package
    # directory with __init__.py. Therefore find_packages() returns [] and
    # must not be used here. py_modules makes `pip install .` install the
    # importable SPHEX modules correctly.
    py_modules=[
        "SPHEX_1_Core",
        "SPHEX_2_Ideal_Surfaces",
        "SPHEX_3_Validation_Suite",
        "SPHEX_4_Run_Validation",
    ],

    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "PyWavelets>=1.1.0",
        "tifffile>=2021.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "sphex=SPHEX_1_Core:main",
            "sphex-validate=SPHEX_3_Validation_Suite:run_full_validation",
        ],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "AFM",
        "atomic force microscopy",
        "biofilm",
        "surface roughness",
        "power spectral density",
        "mathematical biology",
    ],
    zip_safe=False,
)
