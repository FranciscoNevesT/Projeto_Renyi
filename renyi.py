import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import integrate

from numpy import histogram_bin_edges
from bandwidth_estimator import get_bandwidth
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
import time
from scipy.stats import zscore
import warnings
from sklearn.cluster import AgglomerativeClustering

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


def fragment_space(points, max_size_frag = 1000, n_clusters = None):
    p_1 = points[:,-1:]

    if len(p_1) > max_size_frag:
        sample = np.random.randint(0, len(p_1), max_size_frag)
        p_1 = p_1[sample]

    p_1 = np.sort(p_1,axis=0)

    if n_clusters is None:
        n_clusters = int(np.sqrt(len(points)))

    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')

    labels = hierarchical_cluster.fit_predict(p_1)

    bounds = [-np.inf]

    p_last = None
    l_last = labels[0]
    for i in range(len(p_1)):
        p = p_1[i]
        l = labels[i]

        if l != l_last:
            bounds.append(((p+p_last)/2)[0])

        p_last = p
        l_last = l

    bounds = bounds + [np.inf]

    frag = []
    for i in range(len(bounds) - 1):
        bin_start = bounds[i]
        bin_end = bounds[i + 1]

        # Select points within the current bin
        points_t = points[(bin_start < points[:, -1]) & (points[:, -1] < bin_end)]
        frag.append([points_t,bin_start,bin_end])

    return frag


def tau_s_t_multi(args):
    fragment,sample_points,bandwidth,kde_t,size,ref_estimator = args

    points_t, bin_start, bin_end = fragment
    t_val = calc_tau_s_t(points_t, sample_points, bandwidth)

    if ref_estimator == "integral":
        ref = integral_kde(kde_t, [[bin_start, bin_end]], density_function=lambda x: x)
    elif ref_estimator == "proportion":
        ref = len(points_t)/size
    elif ref_estimator == "center":

        if len(points_t) == 0:
            ref = 0
        else:
            if bin_start == -np.inf:
                bin_start = np.min(points_t)

            if bin_end == np.inf:
                bin_end = np.max(points_t)

            center = (bin_start + bin_end) / 2

            p = np.exp(kde_t.score_samples([[center]]))[0]
            dist = (bin_end - bin_start)
            ref = p * dist

    else:
        raise Exception("ref estiamtor not defined")

    return t_val * ref

def tau_s_t(points, sample_points=None, bandwidth="ISJ", num_threads=1,
            max_size_frag = 1000, n_clusters = None, ref_estimator = "proportion"):

    inicio = time.time()
    # Fit kernel density estimation for t dimension
    bandwidth_t = get_bandwidth(points[:, -1].reshape(-1, 1), bandwidth_type=bandwidth)
    kde_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth_t)
    kde_t.fit(points[:, -1].reshape(-1, 1))

    kde_time = time.time()

    points_frag = fragment_space(points, max_size_frag = max_size_frag, n_clusters = n_clusters)

    args = [[i,sample_points,bandwidth,kde_t, len(points), ref_estimator] for i in points_frag]

    frag_time = time.time()

    if num_threads <= 0:
        # Use all available CPU cores if num_threads is negative
        num_threads = None

    if num_threads == 1:
        ts = []

        for arg in args:
            ts.append(tau_s_t_multi(arg))

    else:
        with Pool(processes=num_threads) as pool:
            ts = pool.map(tau_s_t_multi, args)

    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        ts = list(executor.map(calculate_tau_s_t, points_frag))
    """

    end = time.time()

    #print("kde: {} | frag: {} | rest: {}".format(kde_time - inicio, frag_time - kde_time, end - frag_time))

    return np.sum(ts)


def calc_renyi(points, bandwidth="ISJ", num_threads=1, sample_points=None,
               negative_margin = -0.1, view_warning = False, max_size_frag = 1000, n_clusters = None, ref_estimator = "proportion"):

    if points.shape[1] == 2:
        points = zscore(points,axis=0)

    ts = tau_s(points, bandwidth=bandwidth)

    tst = tau_s_t(points, bandwidth=bandwidth, num_threads=num_threads, sample_points = sample_points,
                  max_size_frag = max_size_frag, n_clusters = n_clusters, ref_estimator = ref_estimator)

    metric_renyi = (tst - ts) / tst

    if metric_renyi < 0:
        if metric_renyi < negative_margin and view_warning:
            warnings.warn("Negative value for the metric. Orginal value: {}. Size: {}".format(metric_renyi, len(points)))

        metric_renyi = 0

    return metric_renyi