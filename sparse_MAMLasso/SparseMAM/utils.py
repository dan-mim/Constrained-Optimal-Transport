import numpy as np
import sys
from scipy import spatial

def division_tasks(nb_tasks, pool_size):
    """
    Inputs: (int)
    *nb_tasks
    *pool_size : number of CPU/GPU to divide the tasks between

    Outputs:
    rearranged: numpy list of lists so that rearranged[i] should be treated by CPU[i] (rank=i)
    """
    # The tasks can be equaly divided for each CPUs
    if nb_tasks % pool_size == 0:
        rearranged = np.array([i for i in range(nb_tasks)])
        rearranged = np.split(rearranged, pool_size)

    # Some CPUs will receive more tasks
    else:
        div = nb_tasks // pool_size
        congru = nb_tasks % pool_size
        rearranged1 = np.array([i for i in range(div * congru + congru)])
        rearranged1 = np.split(rearranged1, congru)
        rearranged2 = np.array([i for i in range(div * congru + congru, nb_tasks)])
        rearranged2 = np.split(rearranged2, pool_size - congru)
        rearranged = rearranged1 + rearranged2

    # Output:
    return (rearranged)

# Vectorize function that project vectors onto a simplex, inspired by https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


# Computing the distance matrix:
def distance_matrix(M, S, exact=False):
    """
    This function compute the distance matrix for the exact (free support) problem
    of the inexact problem (fixed support)
    M is the number of considered densities
    S is the size of the probability densities support
    #(R is the size of the barycenter support we look for)
    """
    ### Ancienne methode:
    if not exact:
        R = S
        # eps_R and eps_S are the discretization size of the barycenter image (for R) and the sample images (for S)
        eps_R = 1 / R ** .5
        eps_S = 1 / S ** .5

        la = np.linspace(0, R-1, R)
        lb = np.linspace(0, S-1, S)

        x_R, y_R = la % R ** .5 * eps_R + eps_R / 2, la // R ** .5 * eps_R + eps_R / 2
        x_S, y_S = lb % S ** .5 * eps_S + eps_S / 2, lb // S ** .5 * eps_S + eps_S / 2

        XA = np.array([x_R, y_R]).T
        XB = np.array([x_S, y_S]).T
        M_dist = spatial.distance.cdist(XA, XB, metric='euclidean')

    if exact:
        # Method that respect Bogward theorems:
        K = int(np.round(S ** .5))
        X, Y = np.meshgrid(np.append(np.arange(0, K-1, 1 / M), K-1), np.append(np.arange(0, K-1, 1 / M), K-1))
        ptsK = np.column_stack((X.ravel(), Y.ravel()))

        X, Y = np.meshgrid(np.linspace(0,K-1,K), np.linspace(0,K-1,K))
        ptsk = np.column_stack((X.ravel(), Y.ravel()))

        # Calcul de la distance
        M_dist = spatial.distance.cdist(ptsK, ptsk)

    return(M_dist**2)

def build_M_dist(rank, splitting_work, pool_size, comm, R, S, b, exact=False):
    """
    The distance matrix is build in rank 0 and then it is split with the other processors
    but only the interesting parts of the matrix are shared.
    Then the original distance matrix is delate.
    """
    if rank == 0:
        print('Computing distance matrix...')
        sys.stdout.flush()
        # only one processor build the large distance matrix
        M = len(b)
        M_dist = distance_matrix(M, S, exact) / S  # .astype(np.float16)
        # share the distance matrices between the processors:
        # share_M_btw_procs(M_dist, b, R, splitting_work)
        Mat_dist = {}
        S = {}
        # Sharing the matrix distance to each ranks (but only with the needed columns)
        for rg in range(1, pool_size):
            for m in splitting_work[rg]:
                I = b[m] > 0
                S[m] = np.sum(I)
                Mat_dist[m] = np.reshape(M_dist[:, I], (R * S[m],))
            # sending the matrix
            comm.send(S, dest=rg)
            comm.send(Mat_dist, dest=rg)
            S = {}
            Mat_dist = {}
        # same but direct for rank 0
        for m in splitting_work[0]:
            I = b[m] > 0
            S[m] = np.sum(I)
            Mat_dist[m] = M_dist[:, I]
        # delate the large M_dist
        del M_dist

    # receiving the matrices
    if rank != 0:
        S = comm.recv(source=0)
        Mat_dist = comm.recv(source=0)
        print(rank, S, splitting_work[rank], Mat_dist.keys())
        sys.stdout.flush()
        for m in splitting_work[rank]:
            print(m, S[m])
            sys.stdout.flush()
            Mat_dist[m] = np.reshape(Mat_dist[m], (R, S[m]))

    # Output
    return(Mat_dist)