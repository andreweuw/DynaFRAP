# DynaFRAP

**DynaFRAP** is an interactive tool for semi-automatic analysis of dynamic FRAP experiments in live photobodies.

The software loads a time-resolved FRAP image sequence together with a corresponding neural-network segmentation or log-likelihood output, for example produced by **nnUNet** using [Biom3D](https://biom3d.readthedocs.io). The image sequence and model output are visualized interactively, allowing the user to define a bleached target region and a control region. These regions are then tracked through the remaining frames of the sequence, with manual correction and re-propagation available at any time point.

DynaFRAP is designed to accelerate dynamic FRAP analysis while preserving user control over region selection, tracking corrections, and intensity sampling.

---

## Features

- Interactive visualization of live photobody FRAP time sequences.
- Joint display of image data and neural-network segmentation or log-likelihood outputs.
- Manual selection of target, bleached, and control subspaces.
- Semi-automatic tracking of selected regions through the full time sequence.
- Manual correction of tracked regions at arbitrary time points.
- Re-propagation of corrected tracking results to subsequent frames.
- Parametrization of tracked photobody positions using geometric primitives.
- Mean intensity sampling from tracked subspaces.
- Postprocessing module for experiment-level aggregation, correction, curve fitting, and feature extraction.

Currently the only supported sampling primitives are cylinders with automatically resolved radii.

---

## Main Workflow

The typical DynaFRAP workflow consists of two stages:

1. **Interactive analysis**
   - Load the FRAP time sequence.
   - Load the corresponding segmentation or log-likelihood result.
   - Select target and control regions.
   - Track the selected regions through the sequence.
   - Correct tracking when necessary.
   - Export sampled intensities.

2. **Postprocessing**
   - Aggregate exported intensity measurements across an experiment.
   - Correct target intensities using control intensities.
   - Normalize recovery curves.
   - Fit a one-phase exponential FRAP recovery model.
   - Derive quantitative recovery features such as recovery half-time, mobile fraction, and immobile fraction.

A detailed description of the postprocessing pipeline is available in:
- [Postprocessing overview](docs/postprocessing_overview.pdf)
- [Postprocessing documentation](docs/postprocessing.pdf)


---

## Installation

Clone the repository and install the project dependencies using `uv`:

```bash
git clone <repository-url>
cd <project-directory>
uv sync
```

---

## Interactive FRAP Analysis

To launch the interactive DynaFRAP application, run:

```bash
uv run dynafrap
```

This opens the visualization and interaction interface for selecting, tracking, correcting, and sampling photobody regions across the FRAP time sequence.

---

## Postprocessing

DynaFRAP also provides a postprocessing module for analyzing the generated intensity measurements.

Before running postprocessing, configure the analysis parameters in:

[`src/dynafrap_andreweuw/post_analysis/config.py`](src/dynafrap_andreweuw/post_analysis/config.py)

Run postprocessing with:

```bash
uv run dynafrap-postprocess
```

The postprocessing module performs experiment-level aggregation, photobleaching correction, recovery-curve fitting, and extraction of quantitative FRAP features.

---

## Postprocessing Model Summary

The postprocessing pipeline corrects the measured target intensity using the control intensity:

```text
corrected target intensity = target intensity / control bleaching factor
```

The corrected recovery curves can then be fitted using a one-phase exponential model:

```text
I(t) = I_inf - A exp(-k t)
```

From the fitted model, DynaFRAP derives parameters such as:

* fitted post-bleach intensity,
* recovery plateau,
* recovery amplitude,
* recovery rate constant,
* recovery half-time,
* half-recovery intensity,
* mobile fraction,
* immobile fraction.

For the full mathematical description, see the postprocessing documentation in the `docs/` directory.

---

## Technology Stack

DynaFRAP uses the following core technologies:

* **Biom3D** / **nnUNet** for segmentation or log-likelihood generation.
* **VTK** for visualization, interaction, and 3D region handling.
* **NumPy** and **pandas** for numerical and tabular data processing.
* **SciPy** for recovery-curve fitting.
* **scikit-image** for related scientific Python tools for image processing.

---


## Citation / Acknowledgements

This project uses Biom3D for segmentation-related outputs. See the Biom3D documentation:

```text
https://biom3d.readthedocs.io
```

