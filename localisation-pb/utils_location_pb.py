"""
@daniel.mimouni: --> Working on extensions of MAM adding convex and non convex constraint(s).

I explore a problem of localisation demand/storage.
I have maps of demands for Paris, for a certain amount of time (1 map per months for 12 months for eg or 1 map per day..)
I want to locate storage to reduce my distribution costs.
1) I compute the barycenter of my initial maps to find interesting locations
But I some locations are too expansive so I am restrained to precise area (that I can afford)
Therefore I project my barycenter onto the mask of area where I can afford to settle.
The issue when doing so is that the projected barycenter is not a probability anymore (it does not sum to 1)
2) I include the convex constraint (the location constraint is convex and I know to project onto it) onto the barycenter problem
to find a constraint barycenter (barycenterProjected)
This time the barycenterProjected sums up to 1 ! But also it takes advantages of locations that the previous method did not consider!
3) In fact the location cannot be occupied at 100% due to legislation (laws) then we have to add another convex constraint:
the locations cannot be filled more than stock_max=.009. This is a convex constraint and I know how to project onto it.
The algorithm adapt well again: including the constraint into the optimization scheme (barycenterProjectedConstrained)
is more interesting than applying the constraint after barycenterProjected->barycenterProjected_then_Constrained (does not sum up to 1)
barycenterProjectedConstrained sums up to 1 and explore more locations !
4) To maximize rentability we want to fill storage the maximum we can, therefore we impose that the location is used ONLY if the storage
is greater than stock_min=.004, this is a NON convex constraint, but we know how to project onto it.
If we project after the barycenterProjectedConstrained onto this constraint, the result does not sum up to 1, and including the non convex
constraint into the optimization scheme enables to know what locations are better to fill.

Comments: Incorporating the constraint inside the optimization problem manage better the use of the storage: less locations
are used but they are better used. Meaning projected_barycenter uses more location than barycenterProjected but handle less capacity (store less stuff)
and this is the same for every cases. (see number of pixels used vs sum of the barycenters)

Remark: instead of using a constraint in 2) I could have imposed the support of my barycenter and I would have found the exact constraint barycenter;
#FIXME: compute this exact barycenter to compare results
But for more complex constraints, like 3) (and even more for 4)), this trick is not possible anymore.



"""

from Cmam import MAM, project_onto_stock
from utils_images import *
import sys

def Compute_barycenters(iterations, M, stock_min, stock_max):
    # Load the dataset of 10 spot arrays
    with open("dataset/location_demand", 'rb') as f:
        dataset = pickle.load(f)
    height, width = dataset[0].shape
    for i,data in enumerate(dataset):
        dataset[i] = data.reshape(height*width)
    print('Datas are preprocessed')

    #Classical MAM
    MAM(
        dataset[:M], M_dist=False, exact=False, rho=5, gamma=0,
        keep_track=True, evry_it = 10, name=f'results/MAM_{iterations}i_side{height}_M{M}.pkl',
        visualize=False,
        computation_time=10000, iterations_min=iterations, iterations_max=iterations, precision=10**-12
    )


    # MAM with the projection onto zones
    MAM(
        dataset[:M], M_dist=False, exact=False, rho=5, gamma=0,

        # to deal with a convex constraint
        project=True, shape_map=[height, width],

        keep_track=True, evry_it = 10, name=f'results/MAMProjected_{iterations}i_side{height}_M{M}.pkl',
        visualize=False,
        computation_time=10000, iterations_min=iterations, iterations_max=iterations, precision=10**-12
    )

    # MAM with the projection onto zones with bounds
    stock_max = stock_max /height *40
    MAM(
        dataset[:M], M_dist=False, exact=False, rho=5, gamma=0,

        # to deal with a slightly non-convex constraint
        project=True, shape_map=[height, width], stock_max=stock_max,

        keep_track=True, evry_it=10, name=f'results/MAMProjectedConstrained_{iterations}i_side{height}_M{M}.pkl',
        visualize=False,
        computation_time=100,iterations_min=iterations, iterations_max=iterations, precision=10**-12
    )

    # MAM with projections onto non-convex sets (it's an heuristic)
    stock_min = stock_min /height *40
    MAM(dataset[:M], M_dist=False, exact=False, rho=5, gamma=0,

        # to deal with the non-convex constraint
        project=True, shape_map=[height, width], stock_max=stock_max, stock_min=stock_min,

        keep_track=True, evry_it = 10, name=f'results/MAMProjectedConstrained2_bis_{iterations}i_side{height}_M{M}.pkl',
        visualize=False,
        computation_time=100,iterations_min=iterations, iterations_max=iterations, precision=10**-12)
    return


###################################################
################# PLOTS ###########################
###################################################
def Display_results(barycenter, barycenterProjected, barycenterProjectedConstrained, barycenterProjectedConstrained2,
                    stock_max, stock_min,
                    height):
    print("/!\ BE AWARE that the resulting barycenters have improved resolution using BICUBIC")
    width = height
    # plt.figure(figsize=(12, 12))
    # plt.imshow(barycenter.reshape(height, width), cmap='hot_r', vmax=1.5*stock_max)
    # plt.axis('off')
    # plt.colorbar()
    # plt.title('Crude barycenter')
    barycenter_paris = create_image_nuanced_with_paris_map_increased_resolution(barycenter.reshape(height, width), vmax=1.5*stock_max)
    plt.figure(figsize=(12, 12))
    plt.title('1) Crude barycenter with Paris map', fontsize=20)
    plt.imshow(barycenter_paris)
    plt.axis('off')
    I = barycenter>0
    print(f'1) barycenter has {I.sum()} pixels and sums up to {np.sum(barycenter)}')


    ### 1) Project Wasserstein barycenter after its computation
    projected_barycenter = project_onto_stock(barycenter.reshape(height, width))
    # plt.figure(figsize=(12, 12))
    # plt.title('Projected barycenter')
    # plt.imshow(projected_barycenter, cmap='hot_r', vmax=1.5*stock_max)
    # plt.colorbar()
    projected_barycenter_paris = create_image_nuanced_with_paris_map_increased_resolution(projected_barycenter, vmax=1.5*stock_max)
    plt.figure(figsize=(12, 12))
    plt.title('1&2) Projected barycenter with Paris map', fontsize=20)
    plt.imshow(projected_barycenter_paris)
    plt.axis('off')
    I = projected_barycenter>0
    print(f'1&2) projected_barycenter has {I.sum()} pixels and sums up to {np.sum(projected_barycenter)}')



    ### 2) Take convex constraint inot account in the Barycenter computation
    barycenter_paris = create_image_nuanced_with_paris_map_increased_resolution(barycenterProjected.reshape(height, width), vmax=1.5*stock_max)
    plt.figure(figsize=(12, 12))
    plt.title('2) barycenterProjected with Paris map', fontsize=20)
    plt.imshow(barycenter_paris)
    plt.axis('off')
    I = barycenterProjected>0
    print(f'2) barycenterProjected has {I.sum()} pixels and sums up to {np.sum(barycenterProjected)}')
    # plt.show()
    # FIXME: compute the exact barycenter by fixing the support to compare results with the constraint barycenter.

    ### 3) Take another convex constraint into account: projection after barycenterProjected computation VS taken into account in the barycenter computation
    I = barycenterProjected > stock_max
    barycenterProjected_then_Constrained = barycenterProjected.copy()
    barycenterProjected_then_Constrained[I] = stock_max
    barycenter_paris = create_image_nuanced_with_paris_map_increased_resolution(barycenterProjected_then_Constrained.reshape(height, width), vmax=1.5*stock_max) #*2
    plt.figure(figsize=(12, 12))
    plt.title('3) barycenterProjected_then_Constrained with Paris map (vmax=2*stock_max)', fontsize=20)
    plt.imshow(barycenter_paris)
    plt.axis('off')
    I = barycenterProjected_then_Constrained>0
    print(f'3) barycenterProjected_then_Constrained has {I.sum()} pixels and sums up to {np.sum(barycenterProjected_then_Constrained)}')
    barycenter_paris = create_image_nuanced_with_paris_map_increased_resolution(barycenterProjectedConstrained.reshape(height, width), vmax=1.5*stock_max) #*2
    plt.figure(figsize=(12, 12))
    plt.title('3) barycenterProjectedConstrained with Paris map (vmax=2*stock_max)', fontsize=20)
    plt.imshow(barycenter_paris)
    plt.axis('off')
    I = barycenterProjectedConstrained>0
    print(f'3) barycenterProjectedConstrained has {I.sum()} pixels and sums up to {np.sum(barycenterProjectedConstrained)}')

    ### 4) Take into account a NON convex constraint
    Istock = barycenterProjectedConstrained < stock_min
    barycenterProjectedConstrained_then_2 = barycenterProjectedConstrained.copy()
    barycenterProjectedConstrained_then_2[Istock] = 0
    barycenter_paris = create_image_nuanced_with_paris_map_increased_resolution(barycenterProjectedConstrained_then_2.reshape(height, width), vmax=1.5*stock_max)
    plt.figure(figsize=(12, 12))
    plt.title('4) barycenterProjectedConstrained_then_2 with Paris map', fontsize=20)
    plt.imshow(barycenter_paris)
    plt.axis('off')
    I = barycenterProjectedConstrained_then_2>0
    print(f'4) barycenterProjectedConstrained_then_2 has {I.sum()} pixels and sums up to {np.sum(barycenterProjectedConstrained_then_2)}')

    ###################### Projection 2 #########################
    #############################################################
    I = barycenterProjectedConstrained2>0
    print(f'4) barycenterProjectedConstrained2 before the last projection has {I.sum()} pixels and sums up to {np.sum(barycenterProjectedConstrained2)}')
    # Istock = barycenterProjectedConstrained2 < stock_min
    # barycenterProjectedConstrained2[Istock] = 0
    #############################################################
    barycenter_paris = create_image_nuanced_with_paris_map_increased_resolution(barycenterProjectedConstrained2.reshape(height, width), vmax=1.5*stock_max)
    plt.figure(figsize=(12, 12))
    plt.title('4) barycenterProjectedConstrained 2 with Paris map', fontsize=20)
    plt.imshow(barycenter_paris)
    plt.axis('off')
    I = barycenterProjectedConstrained2>0
    print(f'4) barycenterProjectedConstrained2 has {I.sum()} pixels and sums up to {np.sum(barycenterProjectedConstrained2)}, if these numbers are the same than the previous evaluation \n'
          f' then we don\'t even need to compute the last projection I do by hand, this is good news')
    sys.stdout.flush()
    plt.show()
    return

def load_barycenters(iterations, height, M):
    with open(f"results/MAM_{iterations}i_side{height}_M{M}.pkl", 'rb') as f:
        res = pickle.load(f)
        barycenter = res[0]
    with open(f"results/MAMProjected_{iterations}i_side{height}_M{M}.pkl", 'rb') as f:
        res = pickle.load(f)
        
        barycenterProjected = res[0]
    with open(f"results/MAMProjectedConstrained_{iterations}i_side{height}_M{M}.pkl", 'rb') as f:
        res = pickle.load(f)
        barycenterProjectedConstrained = res[0]
    with open(f"results/MAMProjectedConstrained2_bis_{iterations}i_side{height}_M{M}.pkl", 'rb') as f: #MAMProjectedConstrained2_bis_150i_side100_M12.pkl" , 'rb') as f: #
        res = pickle.load(f)
        barycenterProjectedConstrained2 = res[0]
    return barycenter, barycenterProjected, barycenterProjectedConstrained, barycenterProjectedConstrained2


