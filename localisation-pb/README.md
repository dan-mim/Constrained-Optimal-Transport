
<h1 align="center">📍 Constrained Wasserstein Barycenter — Localization Problem</h1>

<p align="center">
This folder reproduces the numerical results of the paper:<br>
<b>"On the Computation of Constrained Wasserstein Barycenters"</b><br>
Mimouni, de Oliveira, Sempere (2025)<br>
<a href="https://dan-mim.github.io/files/constrained_Wasserstein.pdf">[Read the paper]</a>
</p>

---

## 🧠 Goal

This experiment focuses on a classical **storage localization problem**, where the goal is to compute the **barycenter of spatial demands** under a **stock constraint**.

The barycenter is computed using a constrained version of the **Method of Averaged Marginals (MAM)** with a custom projection operator enforcing feasible support constraints.

---

## ▶️ Running the Demo

To reproduce the experiment:

```bash
cd localization_pb
python demo_localization.py
```

This script:
- Loads multiple input probability measures (e.g. distributions of spatial demand),
- Applies `MAM(..., constraint=project_onto_stock)` using the `Cmam` package,
- Visualizes the constrained barycenter using Paris maps as background.

---

## 🗺️ Pipeline Overview

<p align="center">
  <img src="figs/denoising.PNG" width="1000"/>
</p>

The workflow illustrated above:
1. Starts from noisy or unconstrained barycenter candidates,
2. Applies a projection step to enforce **localization feasibility** (e.g., max number of depots): with convex constraints,
3. With non-convex constraints

---

## 📁 File Description

- `location_pb_execution.py`: Main script that runs the experiment
- `dataset/`: Contains saved spatial distributions
- `Cmam/utils/project_onto_stock.py`: Constraint projection operator, enforcing cardinality or spatial masks
- Dependencies: uses the `Cmam` package found in the root of the repository

---

## 📘 Reference

```bibtex
@article{mimouni2025constrained,
  title={On the Computation of Constrained Wasserstein Barycenters},
  author={Mimouni, Daniel and de Oliveira, Welington and Sempere, Gregorio M.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

---

<p align="center">
For any questions or ideas: feel free to open an issue or fork the repo!<br>
🚀 Happy barycentering!
</p>
