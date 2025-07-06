"""
MAM algorithm to compute CONSTRAINED Wasserstein barycenters
refer to Mimouni, D., de Oliveira, W., & Sempere, G. M.
*ON THE COMPUTATION OF CONSTRAINED WASSERSTEIN BARYCENTERS.*

@author: Daniel Mimouni 2025
"""

from .MAM import MAM
from .utils import project_onto_stock, projection_simplex