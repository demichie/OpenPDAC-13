#!/bin/sh
set -e

# =============================================================================
# Coverage run for OpenPDAC-13/run/testVulcano
# -----------------------------------------------------------------------------
# Purpose
#   This script runs a coverage-oriented version of the testVulcano case for
#   OpenPDAC. It is intended for CI and release-quality assessment workflows.
#
# What it does
#   1. Runs meshing
#   2. Captures a baseline coverage trace
#   3. Runs field initialization (first foamRun)
#   4. Checks whether .gcda files were generated
#   5. Captures coverage for the initialization run
#   6. Resets coverage counters
#   7. Runs a shortened simulation using controlDict.coverage (second foamRun)
#   8. Checks whether .gcda files were generated
#   9. Captures coverage for the simulation run
#  10. Merges the tracefiles, filters external code, and generates HTML
#
# Notes
#   - Post-processing is intentionally skipped.
#   - Coverage capture is focused on applications/OpenPDAC, because this is the
#     location where OpenPDAC coverage files were observed locally.
# =============================================================================

cd "${0%/*}" || exit 1

COVERAGE_ROOT="${COVERAGE_ROOT:-$(pwd)/coverage}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd ../.. && pwd)}"
COVERAGE_DIR="${COVERAGE_DIR:-$PROJECT_ROOT/applications/OpenPDAC}"

mkdir -p "$COVERAGE_ROOT"

echo "==> Project root:   $PROJECT_ROOT"
echo "==> Coverage root:  $COVERAGE_ROOT"
echo "==> Coverage dir:   $COVERAGE_DIR"

# -----------------------------------------------------------------------------
# Helper: print coverage debug information
# -----------------------------------------------------------------------------
print_coverage_debug() {
    local label="$1"

    echo "============================================================"
    echo "Coverage debug: $label"
    echo "Target coverage directory: $COVERAGE_DIR"
    echo "------------------------------------------------------------"

    echo "[gcno] First matches:"
    find "$COVERAGE_DIR" -name "*.gcno" | head -50 || true
    echo "[gcno] Count:"
    find "$COVERAGE_DIR" -name "*.gcno" | wc -l || true

    echo "------------------------------------------------------------"
    echo "[gcda] First matches:"
    find "$COVERAGE_DIR" -name "*.gcda" | head -50 || true
    echo "[gcda] Count:"
    find "$COVERAGE_DIR" -name "*.gcda" | wc -l || true

    echo "============================================================"
}

# -----------------------------------------------------------------------------
# Optional helper: fail clearly if no gcda files are present
# -----------------------------------------------------------------------------
require_gcda_files() {
    local label="$1"
    local count

    count=$(find "$COVERAGE_DIR" -name "*.gcda" | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "ERROR: No .gcda files found in $COVERAGE_DIR after $label"
        echo "This usually means that either:"
        echo "  - coverage instrumentation was not applied correctly, or"
        echo "  - the executed code did not produce runtime coverage files here."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Step 0: show initial debug state
# -----------------------------------------------------------------------------
print_coverage_debug "before meshing"

# -----------------------------------------------------------------------------
# Step 1: meshing
# -----------------------------------------------------------------------------
echo "==> Running meshing"
chmod +x ./01_run_meshing.sh
./01_run_meshing.sh

# -----------------------------------------------------------------------------
# Optional baseline: include instrumented-but-never-executed lines
# -----------------------------------------------------------------------------
echo "==> Capturing baseline coverage"
lcov --capture --initial --directory "$COVERAGE_DIR" --output-file "$COVERAGE_ROOT/coverage_base.info"

print_coverage_debug "after baseline capture"

# -----------------------------------------------------------------------------
# Step 2: field initialization
# -----------------------------------------------------------------------------
echo "==> Running field initialization"
chmod +x ./02_run_fieldInitialization.sh
./02_run_fieldInitialization.sh

print_coverage_debug "after field initialization"

require_gcda_files "field initialization"

echo "==> Capturing coverage after initialization run"
lcov --capture --directory "$COVERAGE_DIR" --output-file "$COVERAGE_ROOT/coverage_init.info"

echo "==> Zeroing counters before short simulation run"
lcov --zerocounters --directory "$COVERAGE_DIR"

print_coverage_debug "after zeroing counters"

# -----------------------------------------------------------------------------
# Step 3: shortened simulation using controlDict.coverage
# -----------------------------------------------------------------------------
echo "==> Preparing short coverage simulation"

cp ./system/controlDict.coverage ./system/controlDict
cp ./system/fvSolution.run ./system/fvSolution
cp ./constant/cloudProperties.run ./constant/cloudProperties

. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

echo "==> Running short simulation with $(getApplication)"
runParallel $(getApplication)

print_coverage_debug "after short simulation"

require_gcda_files "short simulation"

echo "==> Capturing coverage after short simulation run"
lcov --capture --directory "$COVERAGE_DIR" --output-file "$COVERAGE_ROOT/coverage_sim.info"

# -----------------------------------------------------------------------------
# Step 4: merge and filter
# -----------------------------------------------------------------------------
echo "==> Merging tracefiles"
lcov \
  -a "$COVERAGE_ROOT/coverage_base.info" \
  -a "$COVERAGE_ROOT/coverage_init.info" \
  -a "$COVERAGE_ROOT/coverage_sim.info" \
  -o "$COVERAGE_ROOT/coverage_total.info"

echo "==> Filtering external/OpenFOAM files"
lcov \
  --remove "$COVERAGE_ROOT/coverage_total.info" \
  '/opt/openfoam*' \
  '/usr/*' \
  '*/lnInclude/*' \
  --output-file "$COVERAGE_ROOT/coverage_clean.info"

echo "==> Generating HTML report"
genhtml "$COVERAGE_ROOT/coverage_clean.info" --output-directory "$COVERAGE_ROOT/html"

echo "==> Coverage run completed successfully"
echo "HTML report: $COVERAGE_ROOT/html/index.html"
