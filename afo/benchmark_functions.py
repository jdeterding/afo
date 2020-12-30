
import numpy as np


def ackley(x: np.ndarray, a=20, b=0.2, c=2*np.pi)->np.ndarray:
    # TODO: Should be able to pass in any array like object and have this function work.
    # TODO: - Single float values as inputs.  *** is there some numpy to array function? ***
    # TODO: - Any list or list like object.

    """
    Ackleys function is used to test a methods susceptibility to get stuck in local minima.

    f(x) = -a*exp(-b*sqrt(||x||)/d) - exp(sum(cos(c*x))/d) + a + e

    Parameters
    ----------
    x: np.ndarray
        A 2d numpy array with shape (n, d). Where n is the number of points to be evaluated, and d is the number of
        dimentions.
    a: float
        Tunable parameter of the ackley function.
    b: float
        Tunable parameter of the ackley function.
    c: float
        Tunable parameter of the ackley function.

    Examples
    --------
    >>> ackley(np.zeros((1,3)))  # Should be approximately 0.0.
    array([4.4408921e-16])

    >>> ackley(np.zeros((4,3)))  # Should be approximately 0.0.
    array([4.4408921e-16, 4.4408921e-16, 4.4408921e-16, 4.4408921e-16])

    References
    ----------
    M. Kochenderfer and T. Wheeler, "Algorithums for Optimization," The MIT Press,
    pp. 425-426, 2019.

    Returns
    -------
    np.ndarry
        An array with shape (n,).

    """
    d = x.shape[1]
    term_1 = -a*np.exp(-b*np.sqrt(np.sum(x**2, axis=1)/d))
    term_2 = -np.exp(np.sum(np.cos(c*x), axis=1)/d)
    return term_1 + term_2 + a + np.exp(1.0)

"""
if __name__ == "__main__":

    x = np.zeros((3, 2))
    print(ackley(x))
"""
