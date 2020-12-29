
import numpy as np
from typing import List, Iterable, Any, Optional
from numpy.lib import recfunctions as rfs


def full_factorial(*variables: Iterable[Any], variable_names: Optional[List[str]] = None) -> np.recarray:
    """
    TODO: document...
    Parameters
    ----------
    variables: array_like
        An iterable container to define the axies of the full factorial sampling plan. For discrete variables (ints,
        bools, strings) samples are generated at the provided points in the variable axis. For continuious variables
        (float) the provided points are boundries and the sampled points are centered in the boundry.
    variable_names: List of strings
        Variable names to set in the resulting

    Examples
    --------
    >>> full_factorial([2, 4], [True, False])
    rec.array([(2,  True), (4,  True), (2, False), (4, False)],
              dtype=[('f0', '<i4'), ('f1', '?')])

    >>> full_factorial([0.0,2.0,4.0], [True, False], variable_names=["v1", "v2"])
    rec.array([(1.,  True), (3.,  True), (1., False), (3., False)],
              dtype=[('v1', '<f8'), ('v2', '?')])

    Returns
    -------

    """
    variable_arrays = [np.array(vi) for vi in variables]

    for inx in range(len(variable_arrays)):
        if np.issubdtype(variable_arrays[inx].dtype, np.floating):
            variable_arrays[inx] = variable_arrays[inx][:-1] + np.diff(variable_arrays[inx])/2

    raveled_grids = [np.ravel(grd) for grd in np.meshgrid(*variable_arrays)]
    if variable_names:
        dtypes = [(var_name, arr.dtype.name) for arr, var_name in zip(raveled_grids, variable_names)]
        return np.rec.array(raveled_grids, dtype=dtypes)
    else:
        dtypes = ", ".join([arr.dtype.name for arr in raveled_grids])
        return np.rec.array(raveled_grids, dtype=dtypes)


# def stratified_full_factorial():

# def uniform_projection()

# def random_uniform():


if __name__ == "__main__":
    print(full_factorial([0.0, 2.0, 4.0], [True, False]))
    #print(full_factorial([2, 4], [True, False]))
