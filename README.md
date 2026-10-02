# GliCoRe: Coherent Representation Refinement for Glioma Subregion Segmentation

**Official PyTorch Implementation of the Paper:**
> **GliCoRe: Coherent Representation Refinement for Glioma Subregion Segmentation**

 [![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

![GliCoRe overview](docs/Overview+PACER.png)
---

## 🚀 Overview

GliCoRe is a framework for **3-D brain-tumor (glioma) subregion segmentation** from multimodal MRI. Patch-based, spatially-local processing fragments the model's representation in three ways, and GliCoRe restores coherence along each axis:

1. **Context** — a patch carries no information about where it sits in the whole brain;
2. **Appearance** — subregions are spectrally and scale heterogeneous: fine enhancing foci and diffuse edema demand different frequency content;
3. **Structure** — blurred, uncertain boundaries produce holes, breaks, and implausible topology.

The three modules below address these axes in turn.

---

## ✨ Key Contributions

### 1. PACER — Position-Aware Case-context Enhanced Relational Learning
- A **whole-case encoder** builds a low-resolution (64³→8³) memory of the complete scan, supervised by an **occupancy head**.
- **Position-biased cross-attention** injects whole-case context into the 32³ decoder features.
- **Boundary-relation supervision** constrains the interfaces among nested subregions.
- **Training-only**: disabled at inference, so it adds no inference-time cost.

### 2. MEFC — Multi-scale Evidential Frequency Calibration
- **FAEC**: patch-level spectral-energy calibration followed by learned low-/high-frequency response construction on the full-resolution feature.
- **FATB**: temperature-controlled evidence at two decoder scales (`E₂`, `E₃`).
- **FECE**: uncertainty-gated routing of the FAEC responses; fine evidence intervenes only within enhancing-tumor candidates and whole-tumor boundaries.

### 3. UBTR — Uncertainty-Boundary Topology Refinement
- **UBER (Uncertainty-Based Edge Refinement)**: uncertainty-gated injection of spatial + frequency edge cues at uncertain locations.
- **AdaTER**: morphological envelopes and a learnable class-relation matrix rectify subregion topology.

---

## 📂 Project Structure

```text
├── glicore/
│   ├── network_architecture/
│   │   ├── tumor/                  # BraTS: MEFC / UBTR / PACER + GliCoRe model
│   │   ├── synapse/                # Synapse transfer model
│   │   ├── acdc/                   # ACDC transfer model
│   │   └── glicore_generalized.py  # Synapse/ACDC unified model
│   ├── training/
│   │   ├── network_training/       # GliCoReTrainerBraTS/Synapse/ACDC + PACER support
│   │   └── loss_functions/         # evidential loss, PACER occupancy/relation losses
│   ├── inference/                  # sliding-window prediction
│   ├── evaluation/                 # DSC / HD95 / NSD / MASD
│   ├── preprocessing/
│   ├── postprocessing/
│   ├── run/                        # run_training, PACER context cache
│   ├── experiment_planning/
│   ├── dataset_conversion/
│   ├── utilities/
│   └── glicore_config.py           # paper-aligned constants
├── training_scripts/               # train_brats.sh / train_synapse.sh / train_acdc.sh
├── docs/
├── requirements.txt
├── LICENSE
└── README.md
```

Method-to-code map** (key files):

- **PACER** → `glicore/network_architecture/tumor/pacer.py`, `glicore/training/network_training/pacer_training_support.py`
- **MEFC (FAEC + FECE)** → `glicore/network_architecture/tumor/`
- **UBTR (UBER + AdaTER)** → `glicore/network_architecture/tumor/ubtr.py`
- **BraTS model** → `glicore/network_architecture/tumor/glicore_tumor.py`
- **Synapse/ACDC model** → `glicore/network_architecture/glicore_generalized.py`
- **Paper constants** → `glicore/glicore_config.py`

---

## 💻 Environment

The paper environment uses **PyTorch 1.10.2** and a single **NVIDIA RTX 4090**. Install a CUDA-compatible PyTorch build first, then:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

The project follows the nnU-Net-v1 / UNETR++ data convention. Prepare each task with its images, labels, plans, and preprocessed stage, and set `RESULTS_FOLDER` to the output root.
