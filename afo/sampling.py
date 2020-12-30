
import numpy as np
from typing import List, Iterable, Union, Optional
from numpy.lib import recfunctions as rfs


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
    # First convert all iterable variable inputs into numpy arrays.
    variable_arrays = [np.array(vi) for vi in variables]
    # Create mesh grid with matrix indexing resulting grids are in the same order as the input arrays.
    # ie grids.shape == tuple([len(arr) for arr in variable_arrays])
    grids = np.meshgrid(*variable_arrays, indexing="ij")
    # Determine the new shape of the mesh grid after float boundaries have been converted to points.
    org_shape = grids[0].shape 
    new_shape = [i-1 if np.issubdtype(g.dtype, np.floating) else i for g, i in zip(grids, org_shape)]
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
                grids[inx] = grids[inx][low_inxs] + (grids[inx][high_inxs] - grids[inx][low_inxs])/2
        else:
            grids[inx] = grids[inx][high_inxs]  # High inx or low inx don't matter here just need the shape

    raveled_grids = [np.ravel(grid) for grid in grids]

    if variable_names:
        dtypes = [(var_name, arr.dtype.name) for arr, var_name in zip(raveled_grids, variable_names)]
        return np.rec.array(raveled_grids, dtype=dtypes)
    else:
        dtypes = ", ".join([str(arr.dtype) for arr in raveled_grids])
        return np.rec.array(raveled_grids, dtype=dtypes)

# def uniform_projection()

# def random_uniform():


if __name__ == "__main__":
    pass