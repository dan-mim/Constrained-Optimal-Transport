
# 🧭 Optimal Transport Extensions

This repository extends the Method of Averaged Marginals (MAM) to compute **constrained Wasserstein barycenters**, based on:

**Mimouni, D., de Oliveira, W., Sempere, G. M. (2025)**  
*"On the Computation of Constrained Wasserstein Barycenters"*, arXiv ([dan-mim.github.io](https://dan-mim.github.io/files/constrained_Wasserstein.pdf))

⚙️ This includes:
- A **Cmam** package with the function `MAM()` to compute barycenters under convex constraints.
- A working example in `localization_pb/` implementing projection onto a feasible set (`project_onto_stock`) exactly like **Algorithm 1** from the article.
- A research prototype in `sparse_MAMLasso/` for ℓ₁-constrained barycenters (not fully developed, but available for inspiration).

---

## 📦 Installation

```bash
git clone https://github.com/dan-mim/Optimal_Transport_Extensions.git
cd Optimal_Transport_Extensions
pip install -r requirements.txt
```

You may optionally create a Conda environment:

```bash
conda create -n ot_ext python=3.10
conda activate ot_ext
pip install -r requirements.txt
```

---

## 🧠 Algorithm Overview

We extend the classic MAM via **Douglas–Rachford splitting** to solve the constrained problem:

\[
\min_{\mu \in \mathcal{P}} \frac{1}{M}\sum_{m=1}^M W_2^2(\mu, \nu_m)
\quad\text{s.t.}\quad \mu \in X
\]

By discretizing and fixing support, it becomes:

\[
\min_{p \in X, \;\pi^m \ge 0} \sum_{m=1}^M\langle c^m, \pi^m\rangle
\quad\text{s.t.}\;\Pi^m\text{ marginals match }p
\]

**Algorithm 1: Constrained MAM (Douglas–Rachford)**  
```latex
\[
\begin{aligned}
\text{Given } \theta^0,~~\rho > 0.\\
\text{Repeat for }k=0,1,\dots: \\
\quad \pi^{k+1} &= \mathrm{Proj}_{B_X}(\theta^k) \\
\quad \hat{\pi}^{k+1} &= \arg\min_{\pi\in \Pi}
\left\langle c,\pi\right\rangle+\frac{\rho}{2}\|\pi-(2\pi^{k+1}-\theta^{k})\|^2 \\
\quad \theta^{k+1} &= \theta^k + \hat{\pi}^{k+1} - \pi^{k+1}
\end{aligned}
\]
```

- *Proj\_{B\_X}* enforces marginals average into the convex constraint set *X*.
- Matches **Algorithm 1** exactly as implemented in `Cmam/utils/project_onto_stock.py`.

---

## 📁 Repository Structure

```
Optimal_Transport_Extensions/
├── Cmam/
│   ├── __init__.py           # exposes MAM, utils
│   ├── solver.py             # core constrained MAM implementation
│   └── utils/
│       └── project_onto_stock.py
├── localization_pb/          # transport + projection demo
├── sparse_MAMLasso/          # research prototype (ℓ₁ constraints)
├── requirements.txt
└── README.md
```

---

## ▶️ Usage

### 🔹 Example: Localization with stock constraint

```bash
cd localization_pb
python demo_localization.py
```

This demo loads input distributions, applies `MAM(..., constraint=project_onto_stock)`, and visualizes the barycenter under constraints.

### 🔹 ℓ₁-constrained extension

Explore `sparse_MAMLasso/` to see how to add ℓ₁ (sparse) constraints to MAM — work in progress, but a great starting point.

---

## 📘 Citations & Acknowledgements

Primary reference:

```bibtex
@article{mimouni2025constrained,
  title={On the Computation of Constrained Wasserstein Barycenters},
  author={Mimouni, Daniel and de Oliveira, Welington and Sempere, Gregorio M.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

Original MAM paper:

```bibtex
@article{mimouni2024computing,
  title={Computing Wasserstein Barycenter via Operator Splitting: The Method of Averaged Marginals},
  author={Mimouni, Daniel et al.},
  journal={SIAM Journal on Mathematics of Data Science},
  volume={6},
  number={4},
  pages={1000–1026},
  year={2024}
}
```

---

## ✨ Contribution & Exploration

- Feel free to extend MAM to non-convex constraints or GAN-based priors.
- Use `project_onto_stock` and `sparse_MAMLasso/` as templates.
- Pull requests and discussions are welcome — especially for applications in imaging, scenario tree reduction, or structure-constrained OT.
