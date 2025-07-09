# Phreatic Explosion Simulation: Vulcano Island

## 1. Case Summary

This tutorial simulates a volcanic phreatic explosion using the **OpenPDAC** solver, a multiphase flow solver for Particle-Dispersed Clouds.

The workflow demonstrates a sophisticated meshing and initialization process:
1.  **Multi-Block Meshing:** A complex base mesh is generated using `blockMesh`. It defines two distinct regions: an upper "atmospheric" domain with layered refinement near the ground, and a separate, sub-surface "crater" domain.
2.  **Topography-Conforming Deformation:** The upper surface of the mesh is then deformed using `topoGrid`, a custom utility provided with OpenPDAC, to make it conform to the real-world topography of Vulcano Island.
3.  **Atmospheric Initialization:** The simulation first establishes a stable, hydrostatic atmospheric profile across the entire domain.
4.  **Explosion Trigger:** High pressure, temperature, and a gas-particle mixture are initialized specifically within the pre-defined, sub-surface crater blocks. The explosion, therefore, originates from this synthetic crater located beneath the topographic surface.

This case demonstrates advanced techniques like multi-block meshing for local refinement and the use of custom utilities (`topoGrid`) to handle complex geometries.

---

## 2. Case Features

-   **Solver:** `OpenPDAC`
-   **Geometry:** Topography of Vulcano Island, Italy.
-   **Meshing:**
    -   `blockMesh`: Creates a multi-block base mesh defining both the atmospheric domain and a distinct sub-surface crater.
    -   `smoothCraterArea.py`: Custom Python script to prepare topography data.
    -   `topoGrid`: A custom utility from the OpenPDAC developers used to deform the mesh surface to match a real-world topography file.
-   **Initialization:**
    -   A two-stage process using different `controlDict` and `fvSolution` files.
    -   **Stage 1:** Hydrostatic atmospheric profile generation (`hydrostaticInitialisation = true`).
    -   **Stage 2:** Imposing explosion source conditions in the sub-surface crater blocks using `setFields`.
-   **Phenomenon:** Transient, multiphase, explosive dispersal of gas and particles from a sub-surface source.
-   **Workflow:** Features a detailed, step-by-step execution to guide the user through meshing, initialization, the final run, and post-processing.

---

## 3. Prerequisites

### OpenFOAM
-   A working installation of OpenFOAM.
-   The **OpenPDAC** solver and its associated utilities (like `topoGrid`) must be compiled and accessible in your environment.

### Python Environment (Anaconda/Miniconda)
This tutorial uses Python for pre- and post-processing steps. To ensure all dependencies are met, it is **mandatory** to use the provided Conda environment.
1.  **Install Anaconda or Miniconda:** If you don't have it, download and install it from [anaconda.com](https://www.anaconda.com/products/distribution).
2.  **Create the Conda Environment:** Navigate to this tutorial's directory (`testVulcano`) in your terminal and run the following command to create the environment from the `environment.yml` file. This only needs to be done once.
    ```bash
    conda env create -f environment.yml
    ```
    This will create a new environment named `OpenPDACconda` with all the required Python packages.

---

## 4. How to Run the Case

This tutorial is structured with numbered scripts to allow for a step-by-step execution. This is the **recommended approach for learning**.

**Note:** The run scripts (`01_run_meshing.sh`, etc.) will **automatically activate and deactivate** the `OpenPDACconda` environment when they need to execute a Python script. You do not need to activate it manually before running them.

### Step-by-Step Execution (Recommended)

First, make all scripts executable:
```bash
chmod +x Allrun Allclean *.sh
```

#### Step 1: Generate the Mesh
This script handles all meshing operations.
```bash
./01_run_meshing.sh
```
This script will:
1.  Clean the case directory.
2.  Run the Python script to process the topography data.
3.  Create a multi-block base mesh with distinct atmospheric and sub-surface crater regions.
4.  Deform the upper surface of the mesh using the `topoGrid` utility to match the island's topography.
5.  Run `checkMesh` to verify final mesh quality.

**After this step, you should inspect the mesh in ParaView.** Verify that the ground surface is correctly shaped and that the sub-surface crater mesh is present and correctly located.

#### Step 2: Initialize the Flow Fields
This script runs a preliminary simulation to set up a stable atmosphere and then defines the explosion source.
```bash
./02_run_fieldInitialization.sh
```
This script will:
1.  Swap in the `.init` dictionaries, setting `hydrostaticInitialisation` to `true`.
2.  Run the `OpenPDAC` solver to initialize the atmospheric profile.
3.  Use `setFields` to apply high pressure, temperature, and a gas-particle mixture in the sub-surface crater blocks.

**After this step, inspect the initial fields (`p`, `T`, `alpha.solid`) in ParaView** to ensure the atmospheric gradient is correct and the explosion source is confined to the sub-surface crater.

#### Step 3: Run the Main Simulation
This script executes the main phreatic explosion simulation.
```bash
./03_run_simulation.sh
```
This script will:
1.  Swap in the `.run` dictionaries, setting `hydrostaticInitialisation` to `false`.
2.  Run the `OpenPDAC` solver to simulate the explosion dynamics.
3.  Reconstruct the parallel results into a single dataset for visualization.

#### Step 4 (Optional): Run post-processing to generate plots and maps
This script executes the post-processing.
```bash
./04_run_postprocessing.sh
```
This script will:
1.  Create raster maps for pyrocalstic flow invasion.
2.  Create rasper maps for ballistic impact and a csv file with ballistic impact information.
3.  Create videos from different angle of isosurfaces of solid volume fraction.

**After this step, the simulation is complete.** You can also open the case in ParaView to visualize the full time-series of the explosion.

### Automated Execution

To run the entire workflow from start to finish without interruption, simply execute the `Allrun` script:
```bash
./Allrun
```

---

## 5. Cleaning the Case

To remove all generated data (logs, processor directories, time steps) and reset the case to its original state, run the `Allclean` script:
```bash
./Allclean
```
This is essential before starting a new run from scratch.

---

## 6. Description of Key Files

-   **`01_run_meshing.sh` / `02_run_fieldInitialization.sh` / `03_run_simulation.sh`**: The sequential scripts for the workflow.
-   **`Allrun` / `Allclean`**: Master scripts for running everything or cleaning up.
-   **`smoothCraterArea.py`**: A Python script used to pre-process the raw ESRI topography data, likely to smooth the crater area to avoid overly sharp mesh features.
-   **`constant/geometryParameters`**: A supplementary file, included by `blockMeshDict`, that defines the geometric parameters (dimensions, vertices) of the multi-block mesh, particularly for the sub-surface crater.
-   **`system/topoGridDict`**: Configuration for the `topoGrid` utility (from OpenPDAC). It specifies the input topography file for surface deformation and defines the crater's position relative to the topography.
-   **`system/topoSetDict`**: Used by `topoSet` to define cell sets, likely for identifying the sub-surface crater blocks for `setFields`.
-   **`system/controlDict.init` & `system/controlDict.run`**: Different control dictionaries for the initialization and main run phases. The key difference is the `hydrostaticInitialisation` switch.
-   **`system/fvSolution.init` & `system/fvSolution.run`**: May contain different solver settings or tolerances optimized for the quasi-static initialization versus the highly transient main run.
-   **`constant/cloudProperties.init` & `constant/cloudProperties.run`**: Different physical properties for the two stages of the simulation.
-   **`org.0/`**: A backup of the initial condition files. This directory is copied to `0` at the start of the meshing process.
