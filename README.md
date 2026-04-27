# SPHEX

**Spectral Pattern Heterogeneity indeX Analyzer**

*A multiscale framework for quantitative AFM biofilm surface heterogeneity analysis*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.1.1-green.svg)](https://github.com/YourGitHubUsername/SPHEX)

---

## Overview

SPHEX is a Python framework that implements a five-layer hierarchy of
surface descriptors for atomic force microscopy (AFM) height images,
with particular focus on biofilm surface characterization.

The name SPHEX draws a parallel to the digger wasp genus *Sphex*,
known for constructing architecturally complex nests — much like
the structured surface organization of microbial biofilms that
this tool is designed to quantify.

## Five-Layer Architecture

1. **ISO 4287 Roughness Statistics** — Ra, Rq, Rt, Rsk, Rku with
   Fisher-Pearson bias corrections
2. **2D Power Spectral Density** — Hann-windowed, zero-padded FFT
   with directional (CV_Fourier) and radial decompositions
3. **Percentile-based Image Profile (PIP)** — Corrected coefficient
   of variation (CV_PIP) that remains well-defined for zero-mean
   leveled AFM surfaces
4. **Novel Composite Indices**:
   - **Pattern Dominance Index (PDI)** = CV_Fourier / CV_PIP
   - **Heterogeneity Index (HI)** = (Delta_PIP / sqrt(PSD)) × ln(PDI)
5. **Multiscale Decompositions** — Orthogonal probability density
   decomposition (OPD), wavelet transforms, Gaussian-smoothing
   scale sweeps

## Key Features

- Robust to AFM scanning artifacts (spikes, drift, line noise)
- Percentile-based CV that works for zero-mean leveled surfaces
- NIST SRM 2073-targeted self-validation on every run
- Automatic JPK metadata extraction with multi-key fallbacks
- Publication-quality plots and Excel export with units legend
- GUI mode for experimentalists + programmatic API for scripting
- Comprehensive validation suite: **56/56 tests PASS**

## Installation

```bash
git clone https://github.com/YourGitHubUsername/SPHEX.git
cd SPHEX
pip install -r requirements.txt