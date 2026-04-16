#!/bin/sh
set -e

# =============================================================================
# Coverage workflow for OpenPDAC-13/run/testVulcano
# -----------------------------------------------------------------------------
# Purpose
#   This script runs a coverage-oriented version of the testVulcano case.
#
# Strategy
#   - Meshing and topoGrid are executed in parallel, as in the standard setup.
#   - After topoGrid, the decomposed case is reconstructed.
#   - The actual solver runs used for coverage are then executed in serial.
#
# Coverage flow
#   1. Clean case
#   2. Run geometry preparation and meshing
#   3. Decompose and modify the mesh in parallel
#   4. Reconstruct the case after topoGrid
#   5. Run field initialization in serial
#   6. Capture coverage for the initialization run
#   7. Zero counters
#   8. Run a short serial simulation using controlDict.coverage
#   9. Capture coverage for the simulation run
#  10. Merge tracefiles, filter external code, and generate HTML
#
# Notes
#   - Post-processing is intentionally skipped.
#   - Coverage capture is focused on applications/OpenPDAC.
# =============================================================================

cd "${0%/*}" || exit 1

COVERAGE_ROOT="${COVERAGE_ROOT:-$(pwd)/coverage}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd ../.. && pwd)}"
COVERAGE_DIR="${COVERAGE_DIR:-$PROJECT_ROOT/applications/OpenPDAC}"

mkdir -p "$COVERAGE_ROOT"

echo "==> Project root:  $PROJECT_ROOT"
echo "==> Coverage root: $COVERAGE_ROOT"
echo "==> Coverage dir:  $COVERAGE_DIR"

# -----------------------------------------------------------------------------
# Helper: print coverage debug information
# -----------------------------------------------------------------------------
print_coverage_debug() {
    label="$1"

    echo "============================================================"
    echo "Coverage debug: $label"
    echo "Coverage dir: $COVERAGE_DIR"
    echo "------------------------------------------------------------"

    echo "[gcno] First matches:"
    find "$COVERAGE_DIR" -name "*.gcno" | head -50 || true
    echo "[gcno] Count:"
    find "$COVERAGE_DIR" -name "*.gcno" | wc -l || true

    echo "------------------------------------------------------------"
    echo "[gcda] First matches in COVERAGE_DIR:"
    find "$COVERAGE_DIR" -name "*.gcda" | head -50 || true
    echo "[gcda] Count in COVERAGE_DIR:"
    find "$COVERAGE_DIR" -name "*.gcda" | wc -l || true

    echo "------------------------------------------------------------"
    echo "[gcda] First matches in PROJECT_ROOT:"
    find "$PROJECT_ROOT" -name "*.gcda" | head -50 || true
    echo "[gcda] Count in PROJECT_ROOT:"
    find "$PROJECT_ROOT" -name "*.gcda" | wc -l || true

    echo "------------------------------------------------------------"
    echo "[gcda] First matches in \$HOME/OpenFOAM:"
    find "$HOME/OpenFOAM" -name "*.gcda" | head -50 || true
    echo "[gcda] Count in \$HOME/OpenFOAM:"
    find "$HOME/OpenFOAM" -name "*.gcda" | wc -l || true

    echo "============================================================"
}

# -----------------------------------------------------------------------------
# Helper: fail clearly if no gcda files are present in the coverage directory
# -----------------------------------------------------------------------------
require_gcda_files() {
    label="$1"
    count=$(find "$COVERAGE_DIR" -name "*.gcda" | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "ERROR: No .gcda files found in $COVERAGE_DIR after $label"
        echo "This usually means that either:"
        echo "  - the instrumented code was not actually executed, or"
        echo "  - runtime coverage files were written somewhere else."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# OpenFOAM helper functions
# -----------------------------------------------------------------------------
. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

# -----------------------------------------------------------------------------
# Optional Conda activation for Python preprocessing
# -----------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate OpenPDACconda || true
fi

# -----------------------------------------------------------------------------
# Step 0: initial debug
# -----------------------------------------------------------------------------
print_coverage_debug "before meshing"

# -----------------------------------------------------------------------------
# Step 1: clean case
# -----------------------------------------------------------------------------
echo "==> Cleaning case"
chmod +x ./Allclean
./Allclean

# -----------------------------------------------------------------------------
# Step 2: geometry preparation and meshing
# -----------------------------------------------------------------------------
echo "==> Running Python preprocessing"
python3 smoothCraterArea.py > log.smoothCreaterArea

cp ./system/controlDict.init ./system/controlDict

echo "==> Running blockMesh"
runApplication blockMesh

echo "==> Preparing initial fields from org.0"
rm -rf 0
cp -r org.0 0

echo "==> Decomposing case"
runApplication decomposePar

echo "==> Parallel mesh quality check before topoGrid"
runParallel checkMesh -allGeometry -allTopology -writeSets
mv log.checkMesh log.checkMesh0 || true

echo "==> Creating zones"
runParallel createZones

echo "==> Running topoGrid"
runParallel topoGrid

echo "==> Parallel mesh quality check after topoGrid"
runParallel checkMesh -allGeometry -allTopology -writeSets

# -----------------------------------------------------------------------------
# Step 3: reconstruct decomposed case and switch to serial
# -----------------------------------------------------------------------------
echo "==> Reconstructing decomposed case"

# Reconstruct mesh first
runApplication reconstructParMesh -constant

# Reconstruct fields/times if present
runApplication reconstructPar

echo "==> Removing processor directories"
rm -rf processor*

# -----------------------------------------------------------------------------
# Baseline coverage
# -----------------------------------------------------------------------------
echo "==> Capturing baseline coverage"
lcov --capture --initial --directory "$COVERAGE_DIR" --output-file "$COVERAGE_ROOT/coverage_base.info"

print_coverage_debug "after baseline capture"

# -----------------------------------------------------------------------------
# Step 4: field initialization in serial
# -----------------------------------------------------------------------------
echo "==> Preparing case for serial field initialization"
cp ./system/controlDict.init ./system/controlDict
cp ./system/fvSolution.init ./system/fvSolution
cp ./constant/cloudProperties.init ./constant/cloudProperties

echo "==> Running serial field initialization with $(getApplication)"
runApplication $(getApplication)

mv log.foamRun log.foamRun0 || true

echo "==> Running setFields in serial"
runApplication setFields

print_coverage_debug "after serial field initialization"

require_gcda_files "serial field initialization"

echo "==> Capturing coverage after initialization run"
lcov --capture --directory "$COVERAGE_DIR" --output-file "$COVERAGE_ROOT/coverage_init.info"

echo "==> Zeroing counters before short simulation run"
lcov --zerocounters --directory "$COVERAGE_DIR"

print_coverage_debug "after zeroing counters"

# -----------------------------------------------------------------------------
# Step 5: short simulation in serial using controlDict.coverage
# -----------------------------------------------------------------------------
echo "==> Preparing short serial simulation"
cp ./system/controlDict.coverage ./system/controlDict
cp ./system/fvSolution.run ./system/fvSolution
cp ./constant/cloudProperties.run ./constant/cloudProperties

echo "==> Running short serial simulation with $(getApplication)"
runApplication $(getApplication)

print_coverage_debug "after short serial simulation"

require_gcda_files "short serial simulation"

echo "==> Capturing coverage after short simulation run"
lcov --capture --directory "$COVERAGE_DIR" --output-file "$COVERAGE_ROOT/coverage_sim.info"

# -----------------------------------------------------------------------------
# Step 6: merge and filter
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

# -----------------------------------------------------------------------------
echo "==> Coverage run completed successfully"
echo "HTML report: $COVERAGE_ROOT/html/index.html"
