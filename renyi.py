import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import integrate

from numpy import histogram_bin_edges
from bandwidth_estimator import get_bandwidth
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from scipy.stats import zscore
import warnings

def integral_kde(kde,bounds, density_function=lambda x: x):
    """
    Calculate the integral of a given kernel density estimate (KDE) over specified bounds.

    Parameters:
        kde (sklearn.neighbors.KernelDensity): The fitted KDE model.
        bounds (list of tuples): The integration bounds for each dimension of the KDE.
        density_function (callable, optional): A function to apply to the KDE values before integrating.

    Returns:
        float: The integral value.
    """

    def funct(*args):
        point = list(args)
        p = np.exp(kde.score_samples([point]))
        return density_function(p)

    integral_value = integrate.nquad(funct, bounds)[0]
    return integral_value

def tau_s(points, bounds_type = "ref", bandwidth="ISJ"):
    """
    Calculate the tau_s value for Kernel Density Estimation in s dimension.

    Parameters:
        points (array-like): Input data points.
        bounds_type (str, optional): The type of bounds to use for integration.
                                    Can be 'inf' for infinite bounds or 'ref' for data-dependent bounds.
        bandwidth (str, optional): The bandwidth type for KDE estimation. Default is 'ISJ'.

    Returns:
        float: The tau_s value.
    """
    # Fit kernel density estimation for s dimension
    bandwidth_s = get_bandwidth(points[:, :-1].reshape(-1, points.shape[1] - 1),bandwidth_type=bandwidth)
    kde_s = KernelDensity(kernel="gaussian", bandwidth=bandwidth_s)
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


def calc_tau_s_t(points_t, sample_points, bandwidth):
    """
    Estimate the tau_s-t value for Kernel Density Estimation in s-t dimension.

    Parameters:
        points_t (array-like): Input data points in the t dimension.
        sample_points (int): The number of sample points to use for the estimation. If  n   one, the sample points will be set to the number of points in the bin.
        bandwidth (str): The bandwidth type for KDE estimation.

    Returns:
        float: The tau_s-t value, which is a measure of the density estimation in the s-t dimension.
    """

    if len(points_t) <= points_t.shape[1]:
        return 0

    if sample_points is None:
        sample_points = int(len(points_t) * np.log(len(points_t)))

    points_t = points_t[:, :-1]

    # Randomly sample x1 and x2 from points within the bin
    indices = np.random.randint(0, len(points_t), size=(sample_points, 2))
    x1, x2 = points_t[indices[:, 0]], points_t[indices[:, 1]]

    # Calculate w as the difference between x1 and x2
    w = x1 - x2

    # Fit kernel density estimation for s_t dimension
    bandwidth_s_t = get_bandwidth(w, bandwidth_type=bandwidth)
    kde_s_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth_s_t)
    kde_s_t.fit(w)

    # Calculate t_val by evaluating kernel density estimation for s-t dimension at 0 and multiplying by p
    t_val = np.exp(kde_s_t.score_samples([[0] * points_t.shape[1]]))[0]

    return t_val


def fragment_space(points):
    """
    Divide the data space into fragments based on the histogram bin edges.

    Parameters:
        points (array-like): Input data points.

    Returns:
        list: A list of fragments, where each fragment contains data points and its corresponding bin start and end.
    """

    bounds = [-np.inf] +  list(histogram_bin_edges(points[:,-1])) + [np.inf]

    frag = []
    for i in range(len(bounds) - 1):
        bin_start = bounds[i]
        bin_end = bounds[i + 1]

        # Select points within the current bin
        points_t = points[(bin_start < points[:, -1]) & (points[:, -1] < bin_end)]
        frag.append([points_t,bin_start,bin_end])

    return frag


def tau_s_t(points, sample_points=None, bandwidth="ISJ", num_threads=-1):
    """
    Calculate the tau_s-t value for Kernel Density Estimation in t dimension.

    Parameters:
        points (array-like): Input data points.
        sample_points (int, optional): The number of sample points to use for the estimation. If  n   one, the sample points will be set to the number of points in the bin.
        bandwidth (str, optional): The bandwidth type for KDE estimation. Default is 'ISJ'.
        num_threads (int, optional): The number of threads for concurrent execution.
                                     If negative, it will use all available CPU cores.

    Returns:
        float: The tau_s-t value.
    """
    sample_points = len(points)

    # Fit kernel density estimation for t dimension
    bandwidth_t = get_bandwidth(points[:, -1].reshape(-1, 1), bandwidth_type=bandwidth)
    kde_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth_t)
    kde_t.fit(points[:, -1].reshape(-1, 1))

    points_frag = fragment_space(points)

    def calculate_tau_s_t(fragment):
        points_t, bin_start, bin_end = fragment
        t_val = calc_tau_s_t(points_t, sample_points, bandwidth)
        ref = integral_kde(kde_t, [[bin_start, bin_end]], density_function=lambda x: x)
        return t_val * ref

    if num_threads <= 0:
        # Use all available CPU cores if num_threads is negative
        num_threads = None

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        ts = list(executor.map(calculate_tau_s_t, points_frag))

    return np.sum(ts)


def calc_renyi(points, bandwidth="ISJ", num_threads=1):
    """
    Calculate the Renyi divergence metric for Kernel Density Estimation.

    Parameters:
        points (array-like): Input data points.
        bandwidth (str, optional): The bandwidth type for KDE estimation. Default is 'ISJ'.
        num_threads (int, optional): The number of threads for concurrent execution.
                                     If negative, it will use all available CPU cores.

    Returns:
        float: The Renyi divergence metric.
    """


    if points.shape[1] == 2:
        points = zscore(points,axis=0)

    ts = tau_s(points, bandwidth=bandwidth)

    tst = tau_s_t(points, bandwidth=bandwidth, num_threads=num_threads)

    metric_renyi = (tst - ts) / tst

    if metric_renyi < 0:
        #warnings.warn("Negative value for the metric. Orginal value: {}".format(metric_renyi))
        metric_renyi = 0

    return metric_renyi