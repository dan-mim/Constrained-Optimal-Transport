# -*- coding: utf-8 -*-
"""
@author: mimounid
"""

# %% Imports
# Basics
import pickle
import numpy as np
import matplotlib.pyplot as plt
# CPU management:
from mpi4py import MPI
# My codes:
# MAM
from SparseMAM import MAM, projection_simplex


# parallel work parameters:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pool_size = comm.Get_size()

# List of probabilities
N = 3
with open('dataset/b_1_mat.pkl', 'rb') as f:   #b_centers_MNIST #b_1_mat
    l_b = pickle.load(f)
b = l_b[3]
b = b[:N]

iterations = 50
res_lasso = []
l_lasso = np.linspace(0, 2*10**-4, 10)
for lasso in l_lasso : #, 50, 100]
    res_lasso.append(
        MAM(b,lasso=lasso, rho=.1, exact=False, name=f'res_lasso.pkl',
            computation_time=100, iterations_min=iterations, iterations_max=iterations, precision=10 ** -6)
    )

# Taille de la figure globale
num_images = len(res_lasso)
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

for i, results in enumerate(res_lasso):
    bary = results[0]
    lasso = l_lasso[i]
    ax = axes[i] if num_images > 1 else axes
    ax.imshow(bary.reshape(40, 40), cmap='hot_r')
    ax.set_title(f'Lasso {i}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()

