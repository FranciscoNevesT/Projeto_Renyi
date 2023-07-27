import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy import integrate

from numpy import histogram_bin_edges

def integral_kde(kde,bounds, density_function=lambda x: x):
    """
    Calculates the integral of a kernel density estimate (KDE) over a given range.

    Args:
        kde (sklearn.neighbors.KernelDensity): The KDE object representing the density estimate.
        density_function (callable, optional): Function to apply on the density values. Defaults to the identity function.

    Returns:
        float: The calculated integral value.
    """
    def funct(*args):
        """
        Internal function to evaluate the KDE at a given point.

        Args:
            *args: Variable number of arguments representing the point coordinates.

        Returns:
            float: The density value at the given point.
        """
        point = list(args)
        p = np.exp(kde.score_samples([point]))
        return density_function(p)

    integral_value = integrate.nquad(funct, bounds)[0]
    return integral_value

def tau_s(points, bounds_type = "ref", bandwidth="silverman"):
    """
    Calculates tau_s using kernel density estimation.

    Parameters:
    - points: numpy.ndarray, shape (n_samples, n_features)
        The input data points.
    - bandwidth: str or float, optional
        The bandwidth parameter for kernel density estimation. Defaults to "scott".

    Returns:
    - tau_s: float
        The calculated value of tau_s.

    """
    # Fit kernel density estimation for s dimension
    kde_s = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_s.fit(points[:, :-1].reshape(-1, points.shape[1] - 1))

    # Define density function for integration
    density_function = lambda x: x ** 2

    dims = kde_s.n_features_in_
    if bounds_type == "inf":
        bounds = [[-np.inf, np.inf] for _ in range(dims)]
    elif bounds_type == "ref":
        lower_bounds = list(np.min(points[:, :-1], axis=0))
        upper_bounds = list(np.max(points[:, :-1], axis=0))

        bounds = [[lower_bounds[i],upper_bounds[i]] for i in range(len(lower_bounds))]

    # Calculate tau_s using integral_kde function
    return integral_kde(kde_s, density_function=density_function, bounds = bounds)


def calc_tau_s_t(points_t,sample_points,bandwidth):
    # Check if there are points in the bin
    # Todo: make this work when there is 0 elements in the bin
    if len(points_t) == 0:
        # Skip empty bin
        return 0

    points_t = points_t[:, :-1]

    # Randomly sample x1 and x2 from points within the bin
    x1 = points_t[np.random.randint(0, len(points_t), size=(sample_points))]
    x2 = points_t[np.random.randint(0, len(points_t), size=(sample_points))]

    # Calculate w as the difference between x1 and x2
    w = x1 - x2

    # Fit kernel density estimation for s_t dimension
    kde_s_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_s_t.fit(w)

    # Calculate t_val by evaluating kernel density estimation for s-t dimension at 0 and multiplying by p
    t_val = np.exp(kde_s_t.score_samples([[0] * points_t.shape[1]]))[0]

    return t_val


def fragment_space(points):
    bounds = [-np.inf] +  list(histogram_bin_edges(points[:,-1])) + [np.inf]

    frag = []
    for i in range(len(bounds) - 1):
        bin_start = bounds[i]
        bin_end = bounds[i + 1]

        # Select points within the current bin
        points_t = points[(bin_start < points[:, -1]) & (points[:, -1] < bin_end)]

        frag.append([points_t,bin_start,bin_end])

    return frag


def tau_s_t(points,sample_points = None, bandwidth = "silverman"):

    if sample_points is None:
        sample_points = int(points.shape[0] ** 1.5)

    # Fit kernel density estimation for t dimension
    kde_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_t.fit(points[:, -1].reshape(-1, 1))

    points_frag = fragment_space(points)

    ts = []

    for [points_t,bin_start,bin_end] in points_frag:
        t_val = calc_tau_s_t(points_t,sample_points,bandwidth)

        ref = integral_kde(kde_t, [[bin_start,bin_end]], density_function=lambda x: x)

        ts.append(t_val * ref)

    return np.sum(ts)

