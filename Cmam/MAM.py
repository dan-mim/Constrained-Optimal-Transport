# -*- coding: utf-8 -*-
"""
MAM algorithm to compute CONSTRAINED Wasserstein barycenters
refer to Mimouni, D., de Oliveira, W., & Sempere, G. M.
*ON THE COMPUTATION OF CONSTRAINED WASSERSTEIN BARYCENTERS.*

@author: Daniel Mimouni 2025
"""

# Imports

# Basics
import numpy as np
import time
import matplotlib.pyplot as plt
# Import parallel work management:
from mpi4py import MPI
import sys
import pickle
# My codes:
from .utils import division_tasks, build_M_dist, projection_simplex, project_onto_stock



########## MAM algorithm ############
# Optimized Operator Splitting method
def MAM(b, M_dist=False, exact=False, rho=1000, gamma=0,
        project=False, shape_map=[], stock_max=False, stock_min=False,
        keep_track=True, evry_it = 10, name='MAM.pkl', visualize=False, computation_time=100,
                             iterations_min=3, iterations_max=1000, precision=10**-4):
    """
    This version of MAM adresses pb with very large but sparse support. In this case the distance matrix are computed one 
    by one (instead of building a huge distance matrix and only using the sparse rare columns)
    Input:
    *b: (resolution x n) vector collection of n probability distribution.
    *M_dist:( resolution x resolution) is the distance matrix between every supports.
    /!\ Note that M_dist can be a dictionary where M_dist[m] is the distance matrix between the support barycenter and the measure m.
    *exact: (bool) if True we use the free support method (large support considered, refer to Borckwardt), else fixed support, 
    meaning we keep the same support than have the b[i]s for the barycenter.
    *rho: (float) is an hyperparameter that is linked to the convergence speed but has no impact on the solution
    if rho is precised in the arguments of the function then this rho will be used
    * gamma: (float) value of the unbalanced parameter, if gamma=0, MAM treats the balanced barycenter problem
    * keep_track: (bool) should we keep records of the past barycenter approximations and other datas.
    * evry_it: (int) the datas are saved every evry_it iterations
    * name: (str) name of the saved data in the current repository
    * visualize: (bool) if True an approximation of the barycenter is displayed (usefull if image datas are treated)
    * computation_time: (float) maximal computation time in seconds
    * iterations_min: (int) minimal number of iterations
    * iterations_max: (int) maximal number of iterations
    * precision: (float) stopping criteria based on the fixed point method, using the norm max

    Output: p, P, Time, Wdist, l_precisionB, Precision, Iterations, total_time, iterations_k
    * p: (R x 1) the probability distribution of the calculated barycenter
    * P: (R x iterations_k) matrix that store 'p_' after eache iteration of the algorithm
    * Time: (1 x iterations_k) execution time of each iteration
    * Wdist: (1 x iterations_k) evolution of the approximated Wasserstein barycenter distance
    * l_precisionB: (1 x iterations_k) evolution of the evaluation of the accuracy of the approximated Wasserstein barycenter distance
    * Precision: (1 x iterations_k) evolution of the norm max stopping criterion 
    * Iterations: (1 x iterations_k) number of the saved iterations
    * total_time: (float) total computation time
    *iterations_k: (int) number of iterations 
    
    NB: it is possible to save the Pi's: the transport matrices between p and the probability densities

    Infos:
    Refer to the article: "Mimouni, D., Malisani, P., Zhu, J., & de Oliveira, W. (2024). Computing Wasserstein Barycenter
    via operator splitting: the method of averaged marginals. arXiv preprint arXiv:2309.05315."

    (c) Daniel Mimouni 2024
    """""

    st00 = time.time()

    # Parameter initializations
    # Number of probability distributions
    M = len(b)

    # Parallelization initialization:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()
    splitting_work = division_tasks(M, pool_size)

    # Dimension of the pictures
    S = int(b[0].shape[0])

    # Dimension of the support of the targeted barycenter
    # According to Anderes and Borgwardt in 'DiscreteWasserstein barycenters: optimal transport for discrete data'
    # we only need to consider a support size of this size to solve the exact (non convex) pb
    if exact:
        R = ( int(S**.5)*M - M + 1 )**2
        if rank==0:
            print(f'The problem considered is EXACT (non convex barycenter problem with free support), then the suport size is {R}')
            sys.stdout.flush()
    # but if we do not need the exact resolution of the convexe problem we can look for a solution on the same support size as the inital images
    if not exact:
        R = S
        if rank==0:
            print(f'The problem considered is the convex barycenter problem: fixed support of size {R}')
            sys.stdout.flush()

    # I compute the distance matrix for a 2D problem if the distance matrix is not given
    # if a 3D pb is at hand, one should provide the M_dist for e.g.
    try: # see if M_dist is defined
        if type(M_dist) == dict:
            Mat_dist = M_dist
        if type(M_dist) != dict:
            var = M_dist.shape
            for m in splitting_work[rank]:
                I = b[m] > 0
                Mat_dist[m] = M_dist[:, I]
    except: #M_dist is not defined and need to be built (as a matrix then split as a dict)
        Mat_dist = build_M_dist(rank, splitting_work, pool_size, comm, R, S, b, exact)


    # storage of the transport plans
    theta = {}
    S = {}
    sum_theta_mean_local = 0
    inv_sum_S_local = 0
    # PARALLELIZATION
    # iterate over probabilities
    splitting_work = division_tasks(M, pool_size)
    for m in splitting_work[rank]:

        S[m] = np.sum(b[m] > 0)
        inv_sum_S_local = inv_sum_S_local + 1 / S[m]

        # Stored in a dictionnary
        theta[m] = -1 / rho * Mat_dist[m]

        # to compute p
        sum_theta_mean_local = sum_theta_mean_local + np.mean(theta[m], axis=1)

    # Gather and bcast the local solutions
    # Gather and get the sum of the inv_sum_S:
    inv_sum_S = comm.gather(inv_sum_S_local, root=0)
    inv_sum_S = np.sum(inv_sum_S)
    # Bcast:
    inv_sum_S = comm.bcast(inv_sum_S, root=0)
    # Gather and get the sum of the theta_mean's:
    l_sum_theta_mean = comm.gather(sum_theta_mean_local, root=0)
    l_sum_theta_mean = np.array(l_sum_theta_mean)
    sum_theta_mean = np.sum(l_sum_theta_mean, axis=0)
    # Bcast:
    sum_theta_mean = comm.bcast(sum_theta_mean, root=0)

    # probability:
    p = sum_theta_mean / inv_sum_S


    st1 = time.time()
    if rank==0:
        print(f'The computation time of the method starts now, but {np.round(time.time() - st00)}s have been dedicated to initialization (matrix constructions)')
        sys.stdout.flush()


    # Keep track:
    size_saving = iterations_max // evry_it
    if keep_track:
        P = np.zeros((R, size_saving + 1))  # barycentric probability measure
        Time = np.zeros(size_saving + 1)  # time per iteration
        Precision = np.zeros(size_saving + 1)  # precision after each iteration
        Wdist = np.zeros(size_saving + 1)
        l_precisionB = np.zeros(size_saving + 1)
        Iterations = np.zeros(size_saving + 1)

        # Pi = theta.copy()

    # Algorithm iterations:
    spent_time = 0
    iterations_k = 0
    it_save = 0
    while (iterations_k < iterations_min) or (spent_time < computation_time and iterations_k < iterations_max and evol_p>precision): # and count_stop<10):  #  # and evol_p>10**-6) : # and evol_p>10**-16
        iterations_k = iterations_k + 1

        # print('~'*20 +  f" \n Step {k} ...")
        start = time.time()

        # Initialize for inner loops
        sum_theta_mean_local = np.zeros(R)
        if keep_track:
            m_theta = {}
            WD = 0

        # Unbalanced Wasserstein barycenter -> Computing coefficient t:
        t = 1  # if balanced Wasserstein barycenter
        if gamma > 0 and iterations_k > 1:  # if unbalanced
            # Compute t_unbalanced
            coeff_unbalanced_local = 0
            for m in splitting_work[rank]:
                coeff_unbalanced_local = coeff_unbalanced_local + np.linalg.norm(
                    np.sum(theta[m], axis=1) - p) ** 2 / S[m]
            # Gather
            coeff_unbalanced = comm.gather(coeff_unbalanced_local, root=0)
            coeff_unbalanced = np.array(coeff_unbalanced)
            coeff_unbalanced = np.sum(coeff_unbalanced, axis=0)
            # Bcast:
            coeff_unbalanced = comm.bcast(coeff_unbalanced, root=0)

            # Define unbalanced coefficient
            t = min(1, gamma / rho / coeff_unbalanced ** .5)


        # PARALLELIZATION
        # iterate over probabilities
        for m in splitting_work[rank]:
            # index of M_dist I use
            I = b[m] > 0

            # Get the new theta[m]
            # deltaU
            deltaU = ( p - np.sum(theta[m], axis=1) ) / S[m]
            deltaU = np.expand_dims(deltaU, axis=1)
            # W for the projection
            theta[m] = theta[m] - 1 / rho * Mat_dist[m] + 2 * t * deltaU
            # W get normalized before its projection onto the simplex
            theta[m] = theta[m] / b[m][I]
            # the transport plan is un-normalized after the projection onto the simplex
            theta[m] = projection_simplex(theta[m], z=1, axis=0) * b[m][I]

            # Pi[m] = theta[m].copy()
            # this is to evaluate the algorithm advancement
            if keep_track:
                m_theta[m] = np.sum(theta[m], axis=1)
                WD += np.sum( np.multiply(theta[m], Mat_dist[m]) )

            theta[m] = theta[m] - t * deltaU

            # mean of theta:
            sum_theta_mean_local = sum_theta_mean_local + np.mean(theta[m], axis=1) # equivalent to: np.sum(theta[m], axis=1) /S[m]

        # Gather and bcast the local solutions (marginals)
        # Gather and get the sum of the theta_mean's:
        l_sum_theta_mean = comm.gather(sum_theta_mean_local, root=0)
        l_sum_theta_mean = np.array(l_sum_theta_mean)
        sum_theta_mean = np.sum(l_sum_theta_mean, axis=0)
        # Bcast:
        sum_theta_mean = comm.bcast(sum_theta_mean, root=0)

        # Stopping criterion: using the norm max
        evol_p = np.max(np.abs(p - sum_theta_mean / inv_sum_S))

        # Compute the approximated barycenter:
        p = sum_theta_mean / inv_sum_S
        if project:
            height, width = shape_map[0], shape_map[1]
            p = project_onto_stock(p.reshape(height, width)).reshape(height*width)
        if stock_max:
            Istock = p > stock_max
            p[Istock] = stock_max
        if stock_min:
            # projection onto the [stock_min, stock_max]
            # Istock = p < stock_min
            # p[Istock] = 0
            I1 = p < 1/2 * stock_min
            p[I1] = 0
            for r in range(R):
                if p[r] >=1/2 * stock_min and p[r]<stock_min:
                    p[r] = stock_min
            # I2 = p >= 1/2 * stock_min and p<stock_min
            # p[I2] = stock_min



        # Time management: Here is the end of 1 iteration
        end = time.time()
        iteration_time = np.round((end - start), 2)


        #### Keep Track ###
        if rank==0 and iterations_k%evry_it != 0:
            print(f'{iterations_k}: computed in {iteration_time}s; (?)stopping criteria={evol_p}')
            sys.stdout.flush()
        # Save datas
        if keep_track and iterations_k%evry_it == 0:
            # show avancement of the barycenter
            if rank == 0:
                if visualize:
                    plt.close()
                    plt.figure()
                    plt.imshow(np.reshape(p, (int(R**.5),int(R**.5))), cmap='hot_r')
                    plt.colorbar()
                    plt.pause(0.1)

            # Compute some precision variables
            theta1m = np.zeros(R)
            for m in splitting_work[rank]:
                theta1m = theta1m + m_theta[m] / S[m]
            l_theta1m = comm.gather(theta1m, root=0)
            theta1m = np.array(l_theta1m)
            theta1m = np.sum(theta1m, axis=0)
            theta1m = comm.bcast(theta1m, root=0)
            theta1m = theta1m / inv_sum_S
            # Compute the distance of the marginals of theta after the projection onto the simplex, to the subset B:
            distB = 0
            for m in splitting_work[rank]:
                distB += np.linalg.norm(m_theta[m] - theta1m)**2 / S[m]
            l_distB = comm.gather(distB, root=0)
            distB = np.array(l_distB)
            distB = np.sum(distB)
            distB = comm.bcast(distB, root=0)
            distB = distB**0.5

            # Compute the approximated distance:
            l_WD = comm.gather(WD, root=0)
            l_WD = np.array(l_WD)
            WD = np.sum(l_WD)
            # Bcast:
            WD = comm.bcast(WD, root=0) #/ M

            # Fill the matrices
            P[:, it_save] = p
            Time[it_save] = iteration_time
            Precision[it_save] = evol_p
            Wdist[it_save] = WD
            l_precisionB[it_save] = distB
            Iterations[it_save] = iterations_k
            it_save += 1

            # Save results
            if rank==0:
                print(f'{iterations_k}: computed in {iteration_time}s, WD = {WD}, with distB={distB}; (?)stopping criteria={evol_p}')
                sys.stdout.flush()
                # save avancement:
                # save also ? P,
                l_res = [p, P, Time, Wdist, l_precisionB, Precision, Iterations, np.round((end - st1) / 60, 2), iterations_k]
                with open(name, 'wb') as f:
                    pickle.dump(l_res, f)


        # manage time at a global scale:
        spent_time = end - st1
        spent_time = comm.bcast(spent_time , root=0)

    end = time.time()
    total_time = np.round((end - st00) / 60, 2)
    print(f'{iterations_k} iterations in {total_time}min')

    # # Transport plan bcast:
    # list_Pi = []
    # Pi_tot = {} # transport plan
    # for m in splitting_work[rank]:
    #     list_Pi.append((Pi[m],m))
    # list_tot_Pi = comm.gather(list_Pi, root=0)
    # if rank == 0:
    #     for pool in range(pool_size):
    #         for elem in list_tot_Pi[pool]:
    #             m = elem[1]
    #             Pi_m = elem[0]
    #             Pi_tot[m] = Pi_m
    # Pi_tot = comm.bcast(Pi_tot, root=0)

    # Output
    if keep_track:
        return (p, P, Time, Wdist, l_precisionB, Precision, Iterations, total_time, iterations_k)  #Pi_tot
    return(p, total_time, iterations_k)








