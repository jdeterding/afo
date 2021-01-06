
from typing import List, Iterable, Union, Optional, Tuple, Any
import random

import numpy as np


def _full_factorial_mesh_grid(
        *variables: Iterable[Union[str, int, float, bool]],
        stratify: bool = False,
        random_seed: Optional[int] = None) -> List[np.array]:
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


def _ufp_single_axis_index_generator(n: int) -> Iterable[int]:
    """ 
    A generator that returns a random combinations of integer values from 
    [0,n-1]. Once n samples have been generated a new random combination of 
    integer values form [0, n-1].
    """
    indices = []
    while True:
        if len(indices) == 0:
            indices = list(range(n))
            random.shuffle(indices)
            yield indices.pop()
        else:
            yield indices.pop()


def _ufp_index_generator(shape: Tuple[int], random_seed: Optional[int] = None) -> Tuple[Tuple[Any]]:
    """
    
    """
    if random_seed:
        random.seed(random_seed)

    generators = [_ufp_single_axis_index_generator(n) for n in shape]
    return tuple([tuple([next(g) for i in range(max(shape))]) for g in generators])


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


def uniform_projection(
        *variables: Iterable[Union[str, int, float, bool]],
        variable_names: Optional[List[str]] = None,
        stratify: bool = False,
        random_seed: Optional[int] = None) -> np.recarray:
    """

    Parameters
    ----------
    variables
    variable_names
    stratify
    random_seed

    Examples
    --------
    >>> uniform_projection([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], random_seed=1234)
    rec.array([(3, 4), (0, 0), (4, 2), (2, 3), (1, 1)],
              dtype=[('f0', '<i4'), ('f1', '<i4')])

    >>> uniform_projection(np.linspace(0, 1.0, 10), np.linspace(0, 1.0, 5), stratify=True, random_seed=1234)
    rec.array([(0.81298179, 0.09557936), (0.14139918, 0.29607177),
               (0.08726206, 0.97732899), (0.71080029, 0.61952345),
               (0.50034257, 0.14148616), (0.596536  , 0.94799103),
               (0.2619797 , 0.51083102), (0.41252245, 0.37574171),
               (0.96714011, 0.68463076)],
              dtype=[('f0', '<f8'), ('f1', '<f8')])

    Returns
    -------

    """

    grids = _full_factorial_mesh_grid(*variables, stratify=stratify, random_seed=random_seed)
    
    unfp_indices = _ufp_index_generator(grids[0].shape, random_seed=random_seed)
    
    raveled_grids = [np.ravel(grid[unfp_indices]) for grid in grids]
    
    if variable_names:
        dtypes = [(var_name, arr.dtype.name) for arr, var_name in zip(raveled_grids, variable_names)]
        return np.rec.array(raveled_grids, dtype=dtypes)
    else:
        dtypes = ", ".join([str(arr.dtype) for arr in raveled_grids])
        return np.rec.array(raveled_grids, dtype=dtypes)

# def random_uniform():


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_boundaries = np.linspace(0.0, 10, 5)
    y_boundaries = np.linspace(0.0, 10, 7)

    ax = plt.subplot()
    ax.set_xticks(x_boundaries)
    ax.set_yticks(y_boundaries)

    ufp_plan = uniform_projection(x_boundaries, y_boundaries,
                                  variable_names=["x", "y"], stratify=True, random_seed=12345)

    ax.scatter(ufp_plan.x, ufp_plan.y)
    ax.set_xlim((min(x_boundaries)-0.25, max(x_boundaries)+0.25))
    ax.set_ylim((min(y_boundaries) - 0.25, max(y_boundaries) + 0.25))
    plt.grid(True)
    plt.show()
