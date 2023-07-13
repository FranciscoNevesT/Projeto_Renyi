import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy import integrate

def integral_kde(kde, density_function=lambda x: x):
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

    dims = kde.n_features_in_
    bounds = [[-np.inf, np.inf] for _ in range(dims)]

    integral_value = integrate.nquad(funct, bounds)[0]
    return integral_value

def tau_s(points, bandwidth="scott"):
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

    # Calculate tau_s using integral_kde function
    return integral_kde(kde_s, density_function=density_function)

def calc_bins(points, num_points_bins, lower_bounds, upper_bounds):
    """
    Calculates the number of bins for a given set of points.

    Parameters:
    - points: numpy.ndarray
        The input data points.
    - num_points_bins: int
        The desired number of points per bin.
    - lower_bounds: list
        The lower bounds for each dimension.
    - upper_bounds: list
        The upper bounds for each dimension.

    Returns:
    - num_bins: int
        The calculated number of bins.

    """

    t_lower = lower_bounds[-1]
    t_upper = upper_bounds[-1]

    if len(points) // num_points_bins <= 1:
        return 1

    range_bins = [1,len(points)//num_points_bins]

    while range_bins[0] < range_bins[1] - 1:
        bin_m = (range_bins[1] + range_bins[0]) // 2

        d = (t_upper - t_lower) / bin_m

        bin_correct = True
        for t in range(bin_m):
            bin_start = t_lower + t * d
            bin_end = t_lower + (t + 1) * d

            points_t = points[(bin_start < points[:, -1]) & (points[:, -1] < bin_end)]

            if len(points_t) < num_points_bins:
                range_bins = [range_bins[0],bin_m]
                bin_correct = False
                break

        if bin_correct:
            range_bins = [bin_m,range_bins[1]]

    return range_bins[0]


def calc_tau_s_t(points_t,kde_t,sample_points,bandwidth):
    # Check if there are points in the bin
    # Todo: make this work when there is 0 elements in the bin
    if len(points_t) == 0:
        # Skip empty bin
        return 0

    # Calculate p using kernel density estimation for t dimension
    p = np.exp(kde_t.score_samples([[np.mean(points_t[:, -1])]]))[0]
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
    t_val = np.exp(kde_s_t.score_samples([[0] * points_t.shape[1]])) * p

    return t_val


def fragment_space(points, bins=None, num_points_bins=None, lower_bounds=None, upper_bounds=None):
    if lower_bounds is None:
        t_lower = np.min(points[:, -1])
    else:
        t_lower = lower_bounds[-1]

    if upper_bounds is None:
        t_upper = np.max(points[:, -1])
    else:
        t_upper = upper_bounds[-1]

    if bins is None:
        bins = calc_bins(points, num_points_bins, [t_lower], [t_upper])

    d = (t_upper - t_lower) / bins

    frag = []
    for t in range(bins):
        bin_start = t_lower + t * d
        bin_end = t_lower + ((t + 1) * d)

        # Select points within the current bin
        points_t = points[(bin_start < points[:, -1]) & (points[:, -1] < bin_end)]

        frag.append(points_t)

    return frag


def tau_s_t(points, lower_bounds, upper_bounds, bins=None,
            num_points_bins = 10,sample_points = 10000, bandwidth = "scott",
            recursive = False):

    # Fit kernel density estimation for t dimension
    kde_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_t.fit(points[:, -1].reshape(-1, 1))

    if recursive:
        points_frag = []
        points_lista = [points]

        while len(points_lista) > 0:
            p = points_lista.pop(0)

            frags = fragment_space(p, bins=None, num_points_bins=num_points_bins, lower_bounds=None, upper_bounds=None)

            if len(frags) == 1:
                points_frag.append(frags[0])
            else:
                points_lista = points_lista + frags

    else:
        points_frag = fragment_space(points,bins,num_points_bins,lower_bounds,upper_bounds)

    ts = []
    for points_t in points_frag:
        t_val = calc_tau_s_t(points_t,kde_t,sample_points,bandwidth)
        size = np.max(points_t[:,-1]) - np.min(points_t[:,-1])
        ts.append(t_val * size)

    return np.sum(ts)

