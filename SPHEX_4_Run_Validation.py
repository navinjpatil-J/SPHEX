# ============================================================
# SPHEX Validation Launcher
# ============================================================
# Run this to validate the entire SPHEX framework.
# Expected result: 56 PASS | 0 FAIL
#
# Usage: Open in IDLE and press F5
# ============================================================

from SPHEX_3_Validation_Suite import run_full_validation

if __name__ == "__main__":
    df_details, df_summary, df_sweep = run_full_validation()
    print("\nDone. Open validation CSVs to review every check.")
