# SPHEX

**Spectral Pattern Heterogeneity indeX Analyzer**

*A multiscale framework for quantitative analysis of AFM biofilm-surface heterogeneity*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.2.0-green.svg)](https://github.com/navinjpatil-J/SPHEX)

---

## Overview

SPHEX is a Python framework for quantitative analysis of atomic force microscopy (AFM) height maps, with particular emphasis on bacterial-biofilm surface heterogeneity. It combines conventional roughness descriptors with Fourier, percentile-profile, multiscale, wavelet, directional, and self-affine spectral analyses.

The framework is intended to help distinguish surfaces that may have similar average roughness but different spatial organisation, directional structure, or distribution of topographic features across scales.

> **Important:** SPHEX is an image-analysis framework, not a replacement for AFM instrument calibration, appropriate sample preparation, or consistent flattening/artefact correction. Biological interpretation should be supported by raw-image inspection, experimental replication, and independent metadata.

---

## Version 1.2.0

Version 1.2.0 corrects a substantive limitation in the former OPD-derived fractal-dimension output.

### What changed

1. **PSD-slope fractal dimension**

   `Fractal_Dim` is now estimated from the log-log slope of the radially averaged power spectral density (PSD) for a self-affine 2D surface:

   ```text
   PSD(f) ∝ f^(-β)
   Df = 4 − β / 2
   ```

   The previous OPD-derived value could become an image-independent constant because the legacy recursion reduced to uniform PDF scaling. The new PSD-based estimate varies with image structure and is constrained to the physically meaningful graph-surface range of 2–3. A value of `NaN` means that no valid estimate was obtained and should **not** be replaced with zero.

2. **OPD summary fields intentionally withheld**

   The following output columns are now intentionally set to `NaN`:

   ```text
   OPD_Energy_Gap
   OPD_Dominant_Scale
   ```

   The legacy OPD implementation is retained for development/backward compatibility only. Its current energy spectrum is not image-discriminating and must not be used for biological interpretation.

3. **Updated exported metadata and Excel legend**

   Result tables identify the analysis version as `1.2.0`, and the Excel `Units_Legend` explains the revised fractal and OPD outputs.

---

## Analysis architecture

SPHEX provides complementary descriptors rather than a single definition of heterogeneity.

1. **Roughness and height-distribution descriptors**
   - `Ra_nm` — arithmetic mean roughness
   - `Rq_nm` — root-mean-square roughness
   - `Rt_nm` — strict peak-to-valley range
   - `Rt_Robust_nm` — P99.5 − P0.5 robust peak-to-valley range
   - `Rsk` — skewness
   - `Rku` — excess kurtosis as currently implemented

2. **Two-dimensional power spectral density**
   - Hann-windowed, zero-padded 2D FFT
   - radial PSD summary
   - directional Fourier coefficient of variation: `CV_Fourier_pct`

3. **Percentile-based image profile (PIP)**
   - `Delta_PIP_nm = P95 − P5`
   - `CV_PIP_pct` calculated using the central 90% of the height distribution
   - robust Q5 and Q95 descriptors

4. **Composite spectral/height indices**
   - `PDI = CV_Fourier / CV_PIP`
   - `HI = (Delta_PIP / sqrt(Mean_PSD)) × ln(PDI)`
   - `HI_Radial` uses the radial PSD summary

5. **Multiscale and directional descriptors**
   - PSD-slope self-affine fractal dimension
   - wavelet energy entropy: `Wavelet_Complexity_bits`
   - Gaussian-smoothing scale heterogeneity index
   - Fourier anisotropy index and primary direction

---

## Installation

SPHEX v1.2.0 is currently used as a script-based framework. Run commands from the repository root.

```bash
git clone https://github.com/navinjpatil-J/SPHEX.git
cd SPHEX
```

### Recommended virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows Command Prompt**

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Dependencies

```text
numpy
scipy
scikit-image
PyWavelets
tifffile
matplotlib
pandas
openpyxl
```

---

## Input requirements and preprocessing

For scientifically interpretable AFM results:

1. Use a **2D calibrated height image** exported as a floating-point TIFF, preferably a JPK 32-bit float height export.
2. Confirm that height values are in **nanometres**. Integer TIFFs may contain raw detector/count values rather than physical height.
3. Avoid RGB screenshots or rendered false-colour images. They are not raw AFM height data.
4. Apply the same plane/line flattening, filtering, spike policy, and cropping protocol to all images **before** comparative analysis.
5. Use the true physical scan size and image dimensions to calculate pixel pitch:

   ```text
   pixel_size_nm = scan_size_nm / image_width_px
   ```

   For a 10 × 10 µm scan:

   | Image dimensions | Pixel pitch |
   |---:|---:|
   | 512 × 512 px | 19.53125 nm/px |
   | 1024 × 1024 px | 9.765625 nm/px |

6. Always pass both `pixel_size_nm` and `scan_size_nm` to the analysis function. Do not rely on a single fixed pixel-size fallback across images with different resolutions.

---

## Quick start

### Interactive GUI mode

From the repository directory:

```bash
python SPHEX_1_Core.py
```

The GUI allows image selection, metadata review, pixel-size confirmation, analysis, plots, CSV export, and Excel export.

### Programmatic analysis

```python
from pathlib import Path
from SPHEX_1_Core import analyze_afm_image

results = analyze_afm_image(
    image_path="my_afm_height_map.tif",
    pixel_size_nm=19.53125,
    scan_size_nm=10_000.0,
    output_dir=Path("results"),
    save_plots=True,
    save_csv=True,
)

print(results.T)
```

For a 1024 × 1024 image covering the same 10 µm scan, use:

```python
pixel_size_nm = 10_000.0 / 1024.0  # 9.765625 nm/px
```

### Metadata inspection

```python
from SPHEX_1_Core import extract_afm_metadata

metadata = extract_afm_metadata("my_afm_height_map.tif")
print(metadata)
```

Metadata extraction is a convenience feature, not a substitute for checking the AFM acquisition settings. If scan metadata and pixel metadata conflict, verify the original instrument record and derive pixel pitch from the confirmed scan size and image dimensions.

---

## Main output columns

| Output column | Description |
|---|---|
| `Sample_Name` | Image file stem |
| `Image_Path` | Source image path |
| `AFMAnalyzer_Version` | SPHEX analysis version |
| `Scan_Size_nm` | Physical scan size supplied to analysis |
| `Pixel_Size_nm` | Physical pixel pitch supplied to analysis |
| `Ra_nm`, `Rq_nm`, `Rt_nm`, `Rt_Robust_nm` | Roughness and peak-to-valley descriptors |
| `Rsk`, `Rku` | Height-distribution shape descriptors; `Rku` is excess kurtosis in the present implementation |
| `Fractal_Dim` | PSD-slope self-affine fractal estimate; `NaN` means no valid estimate |
| `Mean_PSD_nm2`, `Mean_PSD_Radial_nm2` | Reported 2D and radial PSD summary values |
| `CV_Fourier_pct` | Directional variation in Fourier power |
| `Delta_PIP_nm`, `CV_PIP_pct`, `Q5_nm`, `Q95_nm` | Percentile-based height-profile descriptors |
| `PDI`, `HI`, `HI_Radial` | Composite spectral/height indices |
| `OPD_Energy_Gap`, `OPD_Dominant_Scale` | Intentionally `NaN` in v1.2.0; do not interpret |
| `Wavelet_Complexity_bits` | Shannon entropy of wavelet energy distribution |
| `Scale_Heterogeneity_Index` | Variation of percentile-based CV across smoothing scales |
| `Anisotropy_Index` | Directional imbalance in Fourier power |
| `Primary_Direction_deg` | Binned dominant orientation; interpret only when scan orientation is standardised |
| `Radial_PSD_CSV` | Path to saved radial PSD data, when enabled |

---

## Comparing 512 px and 1024 px images

Correct spatial calibration is necessary but does not automatically make every spectral metric resolution invariant.

For images with the same 10 µm field of view:

```text
512 px Nyquist limit  = 0.0256 cycles/nm
1024 px Nyquist limit = 0.0512 cycles/nm
```

### Recommended practice for group comparisons

- For image-level height metrics such as Ra, Rq, robust Rt, skewness, and percentile-height descriptors, use equal weight per correctly calibrated AFM field.
- Do **not** weight a 1024 px field four times more than a 512 px field simply because it contains four times as many pixels.
- For PSD, radial PSD, Fourier CV, PDI, HI, wavelet, and pixel-scale metrics, either:
  1. anti-alias and downsample 1024 px images to 512 px before analysis; or
  2. calculate all spectral summaries over a common physical frequency range, for example 0.001–0.0256 cycles/nm.
- If several image fields come from one biological biofilm replicate, average technical fields within each biological replicate before calculating group statistics.

---

## Validation

Run the bundled synthetic-surface validation suite from the repository root:

```bash
python SPHEX_4_Run_Validation.py
```

The suite checks expected behaviour on flat, stochastic, sinusoidal, checkerboard, mixed, anisotropic, and fractional-Brownian-motion reference surfaces, together with PDI/HI monotonicity behaviour.

Validation confirms code behaviour on the included synthetic references. It does not replace validation against calibrated AFM standards, independent datasets, or biological replication.

---

## Repository layout

```text
SPHEX_1_Core.py             Main AFM analysis framework
SPHEX_2_Ideal_Surfaces.py   Synthetic reference-surface generators
SPHEX_3_Validation_Suite.py Validation framework
SPHEX_4_Run_Validation.py   Validation launcher
requirements.txt            Python dependencies
setup.py                    Package metadata
LICENSE                     MIT licence
```

---

## Interpretation guidance

- Treat `Ra`, `Rq`, and `Rt_Robust` as complementary measures of vertical amplitude.
- Use strict `Rt` cautiously because a few pixels or rare features can dominate it.
- Interpret `PDI`, `HI`, and `CV_Fourier_pct` together: they are mathematically related, not independent biological tests.
- Interpret anisotropy only after checking scan direction, line artefacts, mounting orientation, and image rotation.
- A lower roughness value does not necessarily mean a less organised surface; a low-Ra image can still have strong directional/spectral structure.
- Inspect raw height maps, line profiles, and radial PSD curves before attributing a numerical difference to EPS, cell packing, substrate effects, or multispecies architecture.

---

## Citation

If you use SPHEX in research, please cite the software repository and the version used for analysis:

> Navinkumar Patil (2025). *SPHEX: Spectral Pattern HEterogeneity indeX Analyzer — A Python framework for multiscale AFM biofilm surface heterogeneity analysis*. Version 1.2.0. https://github.com/navinjpatil-J/SPHEX

Please also cite relevant AFM, PSD, roughness, and experimental biofilm methods appropriate to your study.

---

## License

SPHEX is distributed under the [MIT License](LICENSE).

---

## Contact

**Navinkumar Patil**  
Email: navinjpatil@gmail.com
