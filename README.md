# Online PCA thesis

Code for the first part of the thesis: online estimation of the first principal direction of a 2D data stream, comparing methods that forget old data with methods that remember all of it.

All results come from synthetic data in four regimes: stationary, abrupt change, smooth rotation and random drift. Each experiment is run on 20 seeds.

## How to run

Install the requirements:

    python -m pip install -r requirements.txt

Open the notebooks in Jupyter or VS Code with the same environment. Run each one from a fresh kernel, top to bottom. They do not depend on each other. If you change a .py file, restart the kernel.

Run the tests from the project folder:

    python run_tests.py

## Folders

- methods.py: all the online PCA methods
- data_generators.py: the four synthetic regimes
- metrics.py: axis error, angle error, recovery delay
- benchmark.py: runs the methods on the 20 seeds and saves the results as CSV
- plots.py: tables and plots used in the notebooks
- controllers.py: empty for now, it will hold the HJB and RL gain controllers
- run_tests.py and tests/: unit tests for the methods, metrics, benchmark functions and plots
- notebooks/: the five notebooks described below
- results/: CSV files written by the notebooks, one folder per notebook
- docs/literature_review.md: the sources for all the methods
- archive/: old notebooks and results, not maintained

## Notebooks

In notebooks 01, 02 and 03 every method has its own section, and inside it the four regimes (stationary, abrupt change, smooth rotation, random drift) with the same blocks: a single run, 20 seeds, runtime, plots and repeated measurements. Each notebook ends with basic checks.

01_baselines
- Parameters, common tools, metrics and the four data regimes. These are the settings used by all the other notebooks.
- 1. Cumulative PCA
- 2. Fixed-gain forgetting PCA, covariance update, three gains
- 3. Incremental PCA, rank 1 SVD update
- 4. Rank 1 SVD PCA with a fixed forgetting factor
- Results: one table with all the methods

02_sliding_window
- Part 1, the window methods, each with three window lengths:
  - 1. Exact window, covariance update and downdate
  - 2. Rank 1 SVD window, update and downdate
  - 3. TW-API
- Part 2, analysis:
  - 4. Correctness and numerical drift
  - 5. Truncated downdate and consistent downdate, including when TW-API breaks down
  - 6. Computational cost
  - 7. Window against forgetting at equal noise, and which alpha corresponds to a window of length w
  - 8. Sweep over the window length
- Summary and references

03_oja
- Part 1, the method:
  - 1. Oja PCA with a fixed gain
- Part 2, analysis:
  - 2. Comparable gains for Oja and forgetting PCA
  - 3. Oja against forgetting PCA at equal noise
- Summary and sources

04_sensitivity_analysis
- Part A, robustness to the evaluation protocol:
  - A1. Changing one parameter at a time (threshold, hold, trailing window, warm-up)
  - A2. Which conclusions of 01, 02 and 03 hold for every setting
  - A3. Seed by seed comparison
- Part B, the fixed gain:
  - B1. Sweep over the gain at equal noise
- Part C, the spectral gap:
  - C1. Equal noise at every gap
  - C2. The best gain at each gap
- Summary

05_computational_cost
- 1. Setup: one stationary stream and the three memory levels used in the other notebooks
- 2. Runtime of every method, all measured in the same run
- 3. Results