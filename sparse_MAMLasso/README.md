
<h1 align="center">🔍 Sparse Wasserstein Barycenter (Ongoing Work)</h1>

This folder contains exploratory research on how to compute **sparse Wasserstein barycenters** by modifying the objective function of the classical OT barycenter problem.

Instead of enforcing sparsity through direct constraints on the barycenter support, this approach introduces a novel formulation:  
> the **transport plans themselves** are penalized with an **ℓ₁-norm** regularization.

---

## 📘 Theoretical Context

This work is based on the model and insights discussed in:

📄 `Sparse_barycenter_research.pdf` — available in this folder.

The goal is to encourage sparsity **indirectly**, by promoting sparse coupling matrices (i.e. sparse transport plans), which in turn tend to produce sparser barycenters.

---

## 💾 Dataset

You can download the input dataset from the official MNIST website:  
🔗 [https://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)

---

## ▶️ How to Run

Use the following command to run the modified MAM algorithm:

```bash
python general_execution.py
```

This script:
- Loads input distributions from MNIST,
- Applies the MAM algorithm with ℓ₁-penalized transport plans,
- Uses `SparseMAM`, a drop-in replacement for classical MAM, adapted to include new regularization logic.

---

## 🧱 Code Structure

- `SparseMAM/`: core implementation of the modified algorithm
  - Integrates simplex projection steps
  - Penalizes transport matrices before projecting onto classical OT constraints
- `general_execution.py`: main entry point for experiments
- `Sparse_barycenter_research.pdf`: theoretical reference (ongoing draft)

---

## 🧪 Notes on Implementation

In this formulation:
- Projection onto the **simplex** is explicitly included in the MAM update loop,
- This step is applied **before** the projection onto the constraint set associated with barycenters (e.g., fixed marginals),
- The ℓ₁ regularization is applied directly to each transport plan, promoting sparsity over time.

This is an **experimental setup**, and feedback or contributions are welcome.

---

## 🚧 Status

This is ongoing research, and the code is intended for experimentation and further development.  
Feel free to fork and build upon it — especially if you're exploring structure-inducing priors in OT or barycentric modeling.
