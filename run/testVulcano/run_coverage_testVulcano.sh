#!/bin/sh
set -e

# =============================================================================
# Coverage run for OpenPDAC-13/run/testVulcano
#
# This script:
#   1. runs meshing
#   2. runs field initialization (first foamRun)
#   3. captures coverage for the initialization run
#   4. zeros coverage counters
#   5. runs a shortened simulation using controlDict.coverage (second foamRun)
#   6. captures coverage for the simulation run
#   7. merges the two tracefiles and generates an HTML report
#
# It intentionally skips post-processing.
# =============================================================================

cd "${0%/*}" || exit 1

COVERAGE_ROOT="${COVERAGE_ROOT:-$(pwd)/coverage}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd ../.. && pwd)}"

mkdir -p "$COVERAGE_ROOT"

echo "==> Project root: $PROJECT_ROOT"
echo "==> Coverage output: $COVERAGE_ROOT"

# -----------------------------------------------------------------------------
# Helper: activate conda if available
# -----------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate OpenPDACconda || true
fi

# -----------------------------------------------------------------------------
# Step 0: meshing
# -----------------------------------------------------------------------------
echo "==> Running meshing"
chmod +x ./01_run_meshing.sh
./01_run_meshing.sh

# -----------------------------------------------------------------------------
# Optional baseline: include instrumented-but-never-executed lines
# -----------------------------------------------------------------------------
echo "==> Capturing baseline coverage"
lcov --capture --initial --directory "$PROJECT_ROOT" --output-file "$COVERAGE_ROOT/coverage_base.info"

# -----------------------------------------------------------------------------
# Step 1: field initialization
# -----------------------------------------------------------------------------
echo "==> Running field initialization"
chmod +x ./02_run_fieldInitialization.sh
./02_run_fieldInitialization.sh

echo "==> Capturing coverage after initialization run"
lcov --capture --directory "$PROJECT_ROOT" --output-file "$COVERAGE_ROOT/coverage_init.info"

echo "==> Zeroing counters before short simulation run"
lcov --zerocounters --directory "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Step 2: shortened simulation using controlDict.coverage
# -----------------------------------------------------------------------------
echo "==> Preparing short coverage simulation"

cp ./system/controlDict.coverage ./system/controlDict
cp ./system/fvSolution.run ./system/fvSolution
cp ./constant/cloudProperties.run ./constant/cloudProperties

. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

echo "==> Running short simulation with $(getApplication)"
runParallel $(getApplication)

echo "==> Capturing coverage after short simulation run"
lcov --capture --directory "$PROJECT_ROOT" --output-file "$COVERAGE_ROOT/coverage_sim.info"

# -----------------------------------------------------------------------------
# Step 3: merge and filter
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

echo "==> Coverage run completed"
echo "HTML report: $COVERAGE_ROOT/html/index.html"
