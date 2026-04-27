# ============================================================
# PART 3: Validation Suite v2.0
# ============================================================
# Runs PART 1 (SPHEX_1_Core) on every surface from PART 2,
# compares measured metrics against expected values, and
# performs monotonicity validation on the PDI sweep.
#
# v2.0 — Added: fBm, anisotropic Gaussian, PDI sweep
#         monotonicity check.
#
# Save as: SPHEX_3_Validation_Suite.py
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

# Import from PART 1 and PART 2
from SPHEX_1_Core import (
    calculate_roughness_metrics,
    calculate_psd,
    calculate_fourier_metrics,
    calculate_pip_metrics,
    calculate_pdi,
    calculate_hi,
    calculate_hi_radial,
    multiscale_directional_analysis,
)
from SPHEX_2_Ideal_Surfaces import (
    generate_all_surfaces,
    generate_sweep_surfaces,
    DEFAULT_PIXEL_NM,
)


# ------------------------------------------------------------
# Comparison rule engine
# ------------------------------------------------------------
def check_value(measured, rule):
    """
    Rule formats supported:
      (number, tolerance)        -> within +/- tolerance
      ('greater_than', x)        -> measured > x
      ('less_than', x)           -> measured < x
      ('range', (lo, hi))        -> lo <= measured <= hi
      ('positive', None)         -> measured > 0
      ('negative', None)         -> measured < 0
      ('nan', None)              -> measured is NaN
      ('inf_or_nan', None)       -> measured is NaN or +/- inf
      ('rotation_check', angle)  -> measured close to angle
                                    or angle+90 (mod 180)
    Returns (passed: bool, comment: str).
    """
    # Tuple (number, tolerance)
    if (isinstance(rule[0], (int, float, np.floating))
            and isinstance(rule[1], (int, float, np.floating))):
        target = float(rule[0])
        tol = float(rule[1])
        if not np.isfinite(measured):
            return False, f"got non-finite, expected {target}+/-{tol}"
        ok = abs(measured - target) <= tol
        return ok, f"{measured:.4f} vs {target:.4f}+/-{tol:.4f}"

    kind, arg = rule

    if kind == 'greater_than':
        ok = np.isfinite(measured) and measured > arg
        return ok, f"{measured:.4f} > {arg}?"

    if kind == 'less_than':
        ok = np.isfinite(measured) and measured < arg
        return ok, f"{measured:.4f} < {arg}?"

    if kind == 'range':
        lo, hi = arg
        ok = np.isfinite(measured) and lo <= measured <= hi
        return ok, f"{measured:.4f} in [{lo},{hi}]?"

    if kind == 'positive':
        ok = np.isfinite(measured) and measured > 0
        return ok, f"{measured:.4f} > 0?"

    if kind == 'negative':
        ok = np.isfinite(measured) and measured < 0
        return ok, f"{measured:.4f} < 0?"

    if kind == 'nan':
        return np.isnan(measured), f"isNaN({measured})?"

    if kind == 'inf_or_nan':
        ok = np.isnan(measured) or np.isinf(measured)
        return ok, f"isNaN/Inf({measured})?"

    if kind == 'rotation_check':
        d1 = abs(measured - arg) % 180
        d2 = abs(measured - (arg + 90)) % 180
        d = min(d1, 180 - d1, d2, 180 - d2)
        ok = d < 20.0
        return ok, (f"{measured:.1f} deg vs target {arg} deg "
                     f"(delta={d:.1f} deg)")

    return False, f"unknown rule {kind}"


# ------------------------------------------------------------
# Run full pipeline on a single ideal surface
# ------------------------------------------------------------
def analyze_ideal_surface(image, pixel_size_nm=DEFAULT_PIXEL_NM):
    """Compute all metrics needed for validation."""
    rough = calculate_roughness_metrics(image)
    psd, fx, fy = calculate_psd(image, pixel_size_nm)
    fourier = calculate_fourier_metrics(psd, fx, fy, pixel_size_nm)
    pip = calculate_pip_metrics(image, pixel_size_nm)
    pdi = calculate_pdi(fourier['cv_fourier'], pip['cv_pip'])
    hi = calculate_hi(
        pip['delta_pip'], fourier['mean_psd'],
        fourier['cv_fourier'], pip['cv_pip']
    )
    hi_radial = calculate_hi_radial(
        pip['delta_pip'], fourier['mean_psd_radial'],
        fourier['cv_fourier'], pip['cv_pip']
    )
    ms = multiscale_directional_analysis(image, pixel_size_nm)

    return {
        'Ra_nm':       rough['Ra_nm'],
        'Rq_nm':       rough['Rq_nm'],
        'Rt_nm':       rough['Rt_nm'],
        'Rsk':         rough['Rsk'],
        'Rku':         rough['Rku'],
        'Fractal_Dim': rough['Fractal_Dim'],
        'CV_Fourier_pct': fourier['cv_fourier'],
        'CV_PIP_pct':  pip['cv_pip'],
        'Delta_PIP_nm': pip['delta_pip'],
        'Mean_PSD_nm2': fourier['mean_psd'],
        'Mean_PSD_Radial_nm2': fourier['mean_psd_radial'],
        'PDI':         pdi,
        'HI':          hi,
        'HI_Radial':   hi_radial,
        'Anisotropy_Index':      ms['summary']['anisotropy_index'],
        'Primary_Direction_deg': ms['summary']['primary_direction'],
        'Wavelet_Complexity':    ms['summary']['wavelet_complexity'],
        'Scale_Het_Index':       ms['summary'][
            'scale_heterogeneity_index'],
    }


# ------------------------------------------------------------
# Validate one surface against its expectations
# ------------------------------------------------------------
def validate_surface(image, expected, pixel_size_nm=DEFAULT_PIXEL_NM):
    """Run analysis and check all expected metrics."""
    measured = analyze_ideal_surface(image, pixel_size_nm)
    rows = []
    n_pass = n_fail = 0

    for key, rule in expected.items():
        if key in ('name', 'description',
                   'sweep_alpha', 'sweep_index'):
            continue
        if key not in measured:
            continue
        m = measured[key]
        ok, comment = check_value(m, rule)
        rows.append({
            'Surface':  expected['name'],
            'Metric':   key,
            'Measured': m,
            'Expected': str(rule),
            'Status':   'PASS' if ok else 'FAIL',
            'Comment':  comment,
        })
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    return rows, measured, n_pass, n_fail


# ------------------------------------------------------------
# PDI Sweep monotonicity validation
# ------------------------------------------------------------
def validate_pdi_sweep(sweep_surfaces, sweep_meta,
                       pixel_size_nm=DEFAULT_PIXEL_NM):
    """
    For the PDI sweep, check that both PDI and HI increase
    monotonically as alpha goes from 0 (noise) to 1 (sinusoid).

    A metric is considered monotonically increasing if each
    successive value is >= the previous value minus a small
    tolerance (to allow for sampling noise).
    """
    print("\n" + "=" * 60)
    print("  PDI MONOTONICITY SWEEP VALIDATION")
    print("=" * 60)
    print(f"  {sweep_meta['description']}")
    print(f"  Criterion: PDI and HI must increase with alpha")
    print("-" * 60)

    alphas = []
    pdis = []
    his = []
    cv_pips = []
    cv_fouriers = []
    sweep_rows = []

    for img, exp in sweep_surfaces:
        measured = analyze_ideal_surface(img, pixel_size_nm)
        alpha = exp['sweep_alpha']
        alphas.append(alpha)
        pdis.append(measured['PDI'])
        his.append(measured['HI'])
        cv_pips.append(measured['CV_PIP_pct'])
        cv_fouriers.append(measured['CV_Fourier_pct'])

        print(f"  alpha={alpha:.2f}  "
              f"PDI={measured['PDI']:8.4f}  "
              f"HI={measured['HI']:+8.4f}  "
              f"CV_F={measured['CV_Fourier_pct']:7.2f}%  "
              f"CV_P={measured['CV_PIP_pct']:7.2f}%")

        sweep_rows.append({
            'Surface': exp['name'],
            'Alpha': alpha,
            **{f'M_{k}': v for k, v in measured.items()},
        })

    # Check monotonicity with tolerance for sampling noise
    mono_tol = 0.05  # allow 5% dip relative to range

    # PDI monotonicity
    pdi_range = max(pdis) - min(pdis) if len(pdis) > 1 else 1.0
    pdi_mono = True
    pdi_violations = []
    for i in range(1, len(pdis)):
        if pdis[i] < pdis[i-1] - mono_tol * pdi_range:
            pdi_mono = False
            pdi_violations.append(
                f"alpha {alphas[i-1]:.2f}->{alphas[i]:.2f}: "
                f"PDI {pdis[i-1]:.4f}->{pdis[i]:.4f}"
            )

    # HI monotonicity
    # Filter out NaN values for monotonicity check
    valid_hi = [(a, h) for a, h in zip(alphas, his)
                if np.isfinite(h)]
    hi_mono = True
    hi_violations = []
    if len(valid_hi) >= 2:
        hi_vals = [h for _, h in valid_hi]
        hi_range = max(hi_vals) - min(hi_vals)
        for i in range(1, len(valid_hi)):
            if valid_hi[i][1] < valid_hi[i-1][1] - mono_tol * hi_range:
                hi_mono = False
                hi_violations.append(
                    f"alpha {valid_hi[i-1][0]:.2f}"
                    f"->{valid_hi[i][0]:.2f}: "
                    f"HI {valid_hi[i-1][1]:.4f}"
                    f"->{valid_hi[i][1]:.4f}"
                )

    print("-" * 60)

    # Report PDI monotonicity
    if pdi_mono:
        print("  PDI monotonicity: PASS (increases with alpha)")
    else:
        print("  PDI monotonicity: FAIL")
        for v in pdi_violations:
            print(f"    violation: {v}")

    # Report HI monotonicity
    if hi_mono:
        print("  HI  monotonicity: PASS (increases with alpha)")
    else:
        print("  HI  monotonicity: FAIL")
        for v in hi_violations:
            print(f"    violation: {v}")

    print("=" * 60)

    return (pd.DataFrame(sweep_rows), pdi_mono, hi_mono,
            pdi_violations, hi_violations)


# ------------------------------------------------------------
# Master validation runner
# ------------------------------------------------------------
def run_full_validation(output_dir=None,
                        pixel_size_nm=DEFAULT_PIXEL_NM):
    """Run all validations and save results."""
    if output_dir is None:
        output_dir = (Path.home() / "Desktop" /
                      "AFMAnalyzer_Validation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  AFMAnalyzer Validation Suite v2.0")
    print("  Cross-checking PART 1 against ideal surfaces")
    print("  from PART 2 (10 surfaces + PDI sweep)")
    print("=" * 70)

    # ---- Part A: Individual surfaces (S1-S9) ----
    surfaces = generate_all_surfaces()
    all_rows = []
    summary_rows = []
    total_pass = total_fail = 0

    for img, expected in surfaces:
        print(f"\n>>> {expected['name']}: {expected['description']}")
        rows, measured, np_pass, np_fail = validate_surface(
            img, expected, pixel_size_nm)
        all_rows.extend(rows)

        for r in rows:
            symbol = "+" if r['Status'] == 'PASS' else "X"
            print(f"    {symbol}  {r['Metric']:25s} {r['Comment']}")

        total_pass += np_pass
        total_fail += np_fail

        summary_rows.append({
            'Surface':     expected['name'],
            'Description': expected['description'],
            'Tests_Pass':  np_pass,
            'Tests_Fail':  np_fail,
            **{f'M_{k}': v for k, v in measured.items()},
        })

    # ---- Part B: PDI monotonicity sweep (S10) ----
    sweep_list, sweep_meta = generate_sweep_surfaces()
    (df_sweep, pdi_mono, hi_mono,
     pdi_viol, hi_viol) = validate_pdi_sweep(
        sweep_list, sweep_meta, pixel_size_nm)

    # Count sweep as tests
    if pdi_mono:
        total_pass += 1
    else:
        total_fail += 1
    if hi_mono:
        total_pass += 1
    else:
        total_fail += 1

    # ---- Save outputs ----
    df_details = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(summary_rows)

    details_path = output_dir / "validation_details.csv"
    summary_path = output_dir / "validation_summary.csv"
    sweep_path = output_dir / "validation_pdi_sweep.csv"

    df_details.to_csv(details_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_sweep.to_csv(sweep_path, index=False)

    # ---- Final report ----
    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_pass} PASS  |  "
          f"{total_fail} FAIL  |  "
          f"{total_pass + total_fail} tests")
    print(f"  (includes {len(surfaces)} individual surfaces + "
          f"2 monotonicity checks)")
    print(f"\n  Detailed report:   {details_path}")
    print(f"  Summary report:    {summary_path}")
    print(f"  PDI sweep report:  {sweep_path}")
    print("=" * 70)

    return df_details, df_summary, df_sweep


# ============================================================
if __name__ == "__main__":
    run_full_validation()
