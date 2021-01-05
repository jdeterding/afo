
from typing import List, Iterable, Union, Optional, Tuple
import itertools as its

import numpy as np


def _full_factorial_mesh_grid(
        *variables: Iterable[Union[str, int, float, bool]],
        stratify: bool = False,
        random_seed: Optional[int] = None) -> List[np.array]:
    # TODO: This needs to sort length of variables from long to short.
    # TODO: Throw error if variable not 1d

    # First convert all iterable variable inputs into numpy arrays.
    variable_arrays = [np.array(vi) for vi in variables]
    # Create mesh grid with matrix indexing resulting grids are in the same order as the input arrays.
    # ie grids.shape == tuple([len(arr) for arr in variable_arrays])
    grids = np.meshgrid(*variable_arrays, indexing="ij")
    # Determine the new shape of the mesh grid after float boundaries have been converted to points.
    org_shape = grids[0].shape
    new_shape = [i - 1 if np.issubdtype(g.dtype, np.floating) else i for g, i in zip(grids, org_shape)]
    # Determine how to slice mesh grid into high and low boundary pairs.
    low_slices = tuple([slice(None) if old == new else slice(None, -1) for old, new in zip(org_shape, new_shape)])
    high_slices = tuple([slice(None) if old == new else slice(1, None) for old, new in zip(org_shape, new_shape)])
    # Create bool index to get low and high grid components.
    low_inxs = np.zeros(org_shape, dtype=bool)
    high_inxs = np.zeros(org_shape, dtype=bool)
    low_inxs[low_slices] = True
    high_inxs[high_slices] = True

    if stratify and random_seed:
        np.random.seed(random_seed)

    for inx in range(len(grids)):
        if np.issubsctype(grids[inx].dtype, np.floating):
            if stratify:
                grids[inx] = np.random.uniform(grids[inx][low_inxs], grids[inx][high_inxs])
            else:
                grids[inx] = grids[inx][low_inxs] + (grids[inx][high_inxs] - grids[inx][low_inxs]) / 2
        else:
            grids[inx] = grids[inx][high_inxs]  # High inx or low inx don't matter here just need the shape
        grids[inx] = np.reshape(grids[inx], new_shape)
    return grids


def full_factorial(
        *variables: Iterable[Union[str, int, float, bool]],
        variable_names: Optional[List[str]] = None,
        stratify: bool = False,
        random_seed: Optional[int] = None) -> np.recarray:
    """
    The full factorial, or latin hypercubic, sampling plan samples the parameter space on a full regular grid. The
    sampling plan considers two types of variables, dicrete variables (bools, ints, and strings) and cotinuious
    variables (floats). Discrete variables are sampled at the discrete points specified in the variables inputs.
    Continuious variables specified in the variables containers define the boundries of the latin hyper cube.
    Non-stratifed sampling will sample all of regions defined by the bounries at the center of the the boundry. While
    stratified a sampling plan will pick random point within each of the defined boundry. If f1, f2, ... are the
    continuious varaible axies and d1, d2, ... are the discrete varaible axies when the number of generated points (n)
    is given by.
        n = (len(f1)-1)*(len(f2)-1)*...*len(d1)*len(d2)

    Parameters
    ----------
    variables: array_like
        An iterable container to define the axies of the full factorial sampling plan.
    variable_names: List of strings
        Variable names to set in the resulting record array.
    stratify: bool
        Flage indicating to stratify the continuious axes. Default is False
    random_seed: int
        Set the random seed. Only used if stratify is True. Default is None.

    Examples
    --------
    >>> full_factorial([2, 4], [True, False])
    rec.array([(2,  True), (2, False), (4,  True), (4, False)],
              dtype=[('f0', '<i4'), ('f1', '?')])

    >>> full_factorial([0.0,2.0,4.0], [True, False], variable_names=["v1", "v2"])
    rec.array([(1.,  True), (1., False), (3.,  True), (3., False)],
              dtype=[('v1', '<f8'), ('v2', '?')])

    >>> full_factorial([0.0,1.0,2.0], [2.0,3.0,4.0], stratify=True, random_seed=1234)
    rec.array([(0.19151945, 2.77997581), (0.62210877, 3.27259261),
               (1.43772774, 2.27646426), (1.78535858, 3.80187218)],
              dtype=[('f0', '<f8'), ('f1', '<f8')])

    >>> full_factorial([0.0,1.0,2.0], ['foo', 'bar'], stratify=True, random_seed=1234)
    rec.array([(0.19151945, 'foo'), (0.62210877, 'bar'), (1.43772774, 'foo'),
               (1.78535858, 'bar')],
              dtype=[('f0', '<f8'), ('f1', '<U3')])

    Returns
    -------
    structured array
        A structured array of all the points to evaluate for the sampling plan. Each row is point in the parameter space
        to evaluate.

    """

    grids = _full_factorial_mesh_grid(*variables, stratify=stratify, random_seed=random_seed)

    raveled_grids = [np.ravel(grid) for grid in grids]

    if variable_names:
        dtypes = [(var_name, arr.dtype.name) for arr, var_name in zip(raveled_grids, variable_names)]
        return np.rec.array(raveled_grids, dtype=dtypes)
    else:
        dtypes = ", ".join([str(arr.dtype) for arr in raveled_grids])
        return np.rec.array(raveled_grids, dtype=dtypes)


#def uniform_projection(
#        *variables: Iterable[Union[str, int, float, bool]],
#        variable_names: Optional[List[str]] = None,
#        stratify: bool = False,
#        random_seed: Optional[int] = None) -> np.recarray:

# def random_uniform():

def is_valid_set_of_indices(index: Tuple[int], m: int):
    """
    Determines if a provided set of indices are valid for a uniform projection.
        + Number of unique indices is 0-(m-1).
        + The count of the number of times each indicie appears sattisfies the following,
            1 >= count.max - count.max

    Parameters
    ----------
    index: tuple of ints
        An index into the
    m: int
        Length of the short axis for which indices is being determined.

    Returns
    -------
    bool
        True if valid else false.

    """
    unique, counts = np.unique(index, return_counts=True)
    if len(counts) == m and (counts.max() - counts.min()) <= 1:
        return True
    else:
        return False


def get_short_axis(n, m):
    """ Loop through all possible sets selecting on valid sets. """
    assert n >= m
    return [i for i in its.product(range(m), repeat=n) if is_valid_set_of_indices(i, m)]


def get_2d_ufp_indices(n: int, m: int):
    # TODO: Needs to be randomized.
    # TODO: Should become a generator it takes to long to generate all possible options.
    assert n >= m

    long_axis = list(range(n))
    short_axes = get_short_axis(n, m)

    all_plans = []

    for short_axis in short_axes:
        all_plans.append([(xi, yi) for xi, yi in zip(long_axis, short_axis)])
    return all_plans

"""
Assume that the shape of the axes is sorted from longest to shortest shuch that,
    S = (s_1, s_2, ..., s_i, s_(i-1), ..., s_N)
    where,
        s_i <= s_(i-1)
Then start be determining a valid uniform projection for the first two axes. Note there are two cases. When s_1==s_2 and
when s_1>s_2. The first case we will find is a subset of the second case. So we begin with an example of the second
case.
EX:
      0   1   2   3   4   5 <-- Long Axis
    0-x---|---|---|---x---|
      |   |   |   |   |   |
    1-|---|---x---|---|---|
      |   |   |   |   |   |
    2-|---x---|---|---|---x
      |   |   |   |   |   |
    3-|---|---|---x---|---|
    ^--Short Axis
    Note the following.
        + Along the "Long Axis" all valid indices (0-5) are used once.
        + Along the "Short Axis" the values 0,0,1,2,2,3 are selected for this example.
        + From this we can deduce indices rules for a valid uniform projection.
            + Along both axis each valid index must be used at least once. This is the definition of uniform projection.
            + For the short axis the count (c) of the number of times each index is used should satisfy the following,
                1 >= (c.max - c.min)

Algorithm for multi-dimensional uniform sampling plan.
    - Let n be the number of axes from which to generate the sample plan. (len(shape))
    - If n<2 throw an error at least 2 axes are required.
    - For d1, d2 in zip(shape[::2], shape[1::2])
        - Create & store a generator for the 2d indices of the uniform projection plan.
    - Let G = [g1, g2, ..., gi] be the ith generator of 2d uniform projections plans. 
    - if n%2 == 0 (even number of axes)
        - Start from the last generator
        
    *** Need to re think multi-dim uniform sampling indices. Just because each 2d slice is a uniform projection dosent
    mean the whole thing is. It will be end up far over sampled. *** Need to re think this based on indicies rules. 

"""

if __name__ == "__main__":

    mesh_grid = _full_factorial_mesh_grid([0.0, 1.0, 2.0, 3.0, 4.0],
                                          ["foo", "bar", "baz"],
                                          [True, False])

    print("Mesh Grid:")
    print("Shape:", mesh_grid[0].shape)

    for grid in mesh_grid:
        print(grid)

    sample_plans = get_2d_ufp_indices(4, 3)

    #print("Sample Plans:")
    #for plan in sample_plans:
    #    print(plan)


