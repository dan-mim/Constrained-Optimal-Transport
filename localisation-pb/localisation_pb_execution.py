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

from utils_location_pb import Compute_barycenters, Display_results, load_barycenters

iterations, height, M = 150, 100, 12
stock_max, stock_min = .009, .004
Compute_barycenters(iterations, M, stock_min, stock_max)

stock_max = stock_max / height * 40
stock_min = stock_min / height * 40
Display_results(*load_barycenters(iterations, height, M), stock_max, stock_min, height )