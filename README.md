# survinsights

[![SWH](https://archive.softwareheritage.org/badge/origin/https://pypi.org/project/survinsights//)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://pypi.org/project/survinsights/)
[![PyPI version](https://img.shields.io/pypi/v/survinsights.svg)](https://pypi.org/project/survinsights/)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/survinsights.svg)](https://pypi.org/project/survinsights/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://spdx.org/licenses/MIT.html)

**survinsights** is a Python framework for **interpreting survival models**, with support for **static and longitudinal covariates**, dynamic prediction settings, and Shapley-based explainability methods.

This repository provides **general-purpose tools** for survival model interpretation and also contains **experimental code accompanying specific research papers**.

---

## Overview

`survinsights` is designed to support:
- Model-agnostic explanation of survival predictions
- Longitudinal and time-updated covariates
- Local (individual-level) and global (population-level) explanations
- Dynamic prediction scenarios where explanations evolve over time

The framework can be used independently of any specific publication.

---

## DynSurvX: Experiments from the Paper

This repository **partially accompanies** the paper:

> **DynSurvX: Time-Resolved Explanations for Dynamic Survival Predictions**

Only a **subset of the codebase** is used for this paper.  
In particular, the paper focuses on a Shapley-based explanation method implemented within the `survinsights` framework, referred to as **DynSurvX**.

### What DynSurvX Adds

DynSurvX is an experimental method that:
- Explains survival models with **longitudinal covariates**
- Attributes predictions to both **feature values** and **time intervals**
- Supports **real-time explanations** as new observations become available

These capabilities are implemented on top of the general `survinsights` infrastructure.

---

## Reproducing the Paper Experiments

The following components are used to reproduce the results reported in the DynSurvX paper:

### Relevant Code

- `src/survinsights/local_explaination/_survlongishap/`  
  Core implementation of DynSurvX (longitudinal SHAP explanations)

- `src/survinsights/longi_prediction.py`  
  Utilities for dynamic survival prediction

### Experimental Scripts

The DynSurvX experiments currently support **18 configurations**, combining:

- **3 datasets:** Simulation (`simu`), PBC-Seq (`PBC`), and MSK-Chord (`MSK`).
- **3 models:** CoxSig (`CoxSig`), Dynamic DeepHit (`DDH`), and DySurv (`DySurv`).
- **2 prediction functions:** Expected remaining survival time (`survival`) and median residual survival time (`med-survival`).

Experiments can be run from the terminal using the unified script:

```bash
python scripts/longisurvshap_unified.py \
    --dataset PBC \
    --model CoxSig \
    --prediction-function med-survival
```

Hyperparameters for each configuration are defined in `longisurvshap_experiments.json` and automatically loaded based on the selected dataset, model, and prediction function.

The original `(dataset)_(mean|med)_(model).py` scripts are also available for individual experiments.

### Figure Reproduction

- `(dataset)_figure_reproduce/*.ipynb`  
  Jupyter notebooks to reproduce figures reported in the paper

---

## Datasets Used in the Paper

### Real Clinical Datasets

- **PBCseq**  
  Publicly available via the R package **JMbayes**

- **MSK-Chord**  
  Available through the [cBioPortal platform](https://www.cbioportal.org/study/clinicalData?id=msk_chord_2024)  
  (data access approval required)

### Simulated Data

Simulated datasets are included in the repository and are used to:
- Validate temporal attribution accuracy
- Study robustness to prediction targets
- Provide ground-truth temporal importance patterns

---

## Tutorials and Examples

Beyond the DynSurvX paper, the **tutorial notebooks** `tutorial_(dataset)_(mean|med)_(model).ipynb` demonstrating how to use `survinsights` for:
- Longitudinal survival prediction
- Feature attribution
- Dynamic explanations

---

## Installation

```bash
virtualenv -p python3.12 .venv_survinsights
source .venv_survinsights/bin/activate
pip install -r pip_requirements.txt
