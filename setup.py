from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SPHEX",
    version="1.1.1",
    author="Navinkumar Patil",
    author_email="navinjpatil@gmail.com",
    description=(
        "SPHEX: Spectral Pattern Heterogeneity indeX Analyzer — "
        "A multiscale framework for quantitative AFM biofilm "
        "surface heterogeneity analysis"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/navinjpatil-J/SPHEX",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
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
)
