# Energy Efficiency and Causal Irreducibility in Reentrant SNNs

This repository contains the official simulation framework, optimized data structures, and plotting tools supporting the research paper: **"Energy Efficiency and Causal Irreducibility: An Analysis of Integrated Information ($\Phi_{\text{proxy}}$) in Reentrant Neuromorphic Architectures"** (Applied Sciences, MDPI).

## 📊 Summary of Experimental Results
Compared to conventional linear architectures, the proposed Reentrant Spiking Neural Network (SNN) achieves sparse temporal coding driven by recurrent lateral inhibition, resulting in:
* **Energy Savings:** **60.8% net reduction** in dynamic power consumption ($0.19991\ \mu\text{J}$ vs $0.51020\ \mu\text{J}$).
* **Physical Sparsity:** Mitigated redundant firing activity to only **3,983 total spikes**.
* **Enhanced Performance:** Achieved an **Acoustic Accuracy of 89.40%** and an **F1-Score of 0.882** on the Google Speech Commands v2 dataset.
* **Structural Stability:** Maintained a cohesive thermodynamic state ($S = 0.1503\text{ Hartleys}$) well below the 37% universal entropy collapse threshold ($1/e$).

## 🛠️ Repository Architecture
* `calculo_phi_corrigido.py`: Computes the non-negative multivariate causal integration proxy ($\Phi_{\text{proxy}} = 2.6364$) via Kullback-Leibler Divergence.
* `gerar_figura1_estabilidade.py`: Plots Carvalho Rodrigues' Structural Stability Curve.
* `gerar_figura2_aprendizagem.py`: Replicates the Cross-Entropy loss decay over 50 optimization epochs.
* `gerar_figura3_raster.py`: Simulates the sparse, asynchronous neuromorphic raster plot of 3,983 spikes under reentrant inhibition.
* `gerar_figura4_tradeoff.py`: Visualizes the Pareto frontier and the metric-energy optimization trade-offs.
* `gerar_figura5_comparativo.py`: Draws the final macroscopic bar-and-line comparative analysis (MDPI format).

## 🚀 Environment & Compatibility Shield
To guarantee exact data provenance and long-term script reproducibility, all execution paths are pinned under a **"compatibility shield"** requiring:
* `python >= 3.8`
* `numpy < 2.0` (Strict backward compatibility layer for matrix operators)
* `matplotlib`
* `scipy`

## 🗄️ Static Citation and Provenance
A permanent snapshot of this computational environment, including pre-trained weights and compiled logs, is frozen under Zenodo:
**DOI:** [https://doi.org](https://doi.org)
