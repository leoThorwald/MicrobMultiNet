# MicrobMultiNet 🦠

> **Neural ODE-based framework for microbial community dynamics inference**  

---

## Overview

MicrobMultiNet is a neural ODE-based framework for modeling and inferring interaction networks in microbial time series data. It is designed as a probabilistic generalized Lotka–Volterra (gLV) model for gut microbiome dynamics.

The core idea is to replace the restrictive parametric assumptions of gLV with flexible neural architectures while preserving biological interpretability — specifically, the ability to extract **who interacts with whom** in the microbial community.

For full results, model derivations, and benchmarking details, see the [**project report**](report.pdf).

---

## Repository Structure

```
.
├── report.pdf                  # Full lab immersion report
├── environment.yml             # Conda environment file
├── training_demo.ipynb         # Jupyter notebook: training loop & model selection
├── models.py                   # MicrobMultiNet and HybridODE model 
└── data/                       # Synthetic dataset
```
---

## Models

### 🔬 MicrobMultiNet *(Main Contribution)*

MicrobMultiNet is a neural ODE architecture specifically designed for microbial interaction inference. It consists of **N parallel neural networks** — one per microbial species — whose inputs are gated by a **learnable interaction mask matrix** M ∈ ℝᴺˣᴺ.

**Key design principles:**

- Each species `i` has a mask vector **mᵢ** ∈ ℝᴺ that learns which other species influence its growth rate
- If `mᵢⱼ ≈ 0`, species `j` has negligible effect on species `i` — enabling direct interaction network extraction
- The ODE right-hand side takes the multiplicative form `dyᵢ/dt = yᵢ · gᵢ`, ensuring non-negativity and biological plausibility
- L1 regularization on the mask encourages sparse, interpretable interaction networks

**Architecture summary:**

```
For each species i:
  xᵢ = mᵢ ⊙ y                    # masked community state
  gᵢ = MLPᵢ(xᵢ; θᵢ) : ℝᴺ → ℝ   # species-specific growth modifier
  dyᵢ/dt = yᵢ · gᵢ               # ODE right-hand side

Full system:
  dy/dt = y ⊙ g(y; Θ, M)
```

See [Section 6.3 of the report](report.pdf) for the full mathematical derivation.

---

### ⚙️ HybridODE *(Modular Baseline)*

HybridODE is a flexible, modular ODE model with three interchangeable components. Parts can be added or removed to configure different model variants:

| Module | Description | Markovian? |
|--------|-------------|------------|
| **Part 1 — Pure Lotka-Volterra** | Classic parametric gLV dynamics. Full interpretability, limited expressivity. | ✅ Yes |
| **Part 2 — Markovian Neural Network** | Input and output dimensions equal the number of species (N → N). Standard neural ODE over the observed state space. | ✅ Yes |
| **Part 3 — Non-Markovian Neural Network** | Augmented input and output dimensions beyond N. Allows the model to carry latent hidden state, capturing memory effects and more complex dynamics. | ❌ No |

The modularity allows controlled ablation: you can swap between a purely mechanistic model, a standard neural ODE, and an augmented latent-state model to study the trade-off between interpretability and expressivity.

---

## Getting Started

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate microbmultinet
```

### 2. Run the training demo

Open training_demo.ipynb to configure and run the training loop. 

The notebook lets you:
- Choose between **MicrobMultiNet** and **HybridODE**
- Configure HybridODE modules (toggle Part 1 / 2 / 3)
- Visualize predicted trajectories and extracted interaction matrices

---


## Background & Motivation

The human gut microbiome consists of trillions of microorganisms whose dynamics are shaped by diet, antibiotics, and host physiology. Understanding *who interacts with whom* is critical for predicting responses to interventions and linking microbial composition to disease.

MicrobMultiNet addresses these limitations by learning a continuous-time vector field parameterized by neural networks, while using structural constraints (the mask matrix + L1 regularization) to retain interpretability.

See [Section 2–3 of the report](report.pdf) for a detailed discussion.



---

## Acknowledgements

This project builds on the MDSINE2 dataset and synthetic data generation methodology from:

> Gibson et al., *Learning ecosystem-scale dynamics from microbiome data with MDSINE2*, Nature Microbiology, 2025.

Neural ODE framework based on:

> Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018.