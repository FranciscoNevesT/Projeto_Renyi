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

    Args:
        points (numpy.ndarray): Input data points.
        bandwidth (float, optional): Bandwidth for kernel density estimation. Defaults to 0.5.

    Returns:
        float: tau_s value.
    """
    # Fit kernel density estimation for s dimension
    kde_s = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_s.fit(points[:, :-1].reshape(-1, points.shape[1] - 1))

    # Define density function for integration
    density_function = lambda x: x ** 2

    # Calculate tau_s using integral_kde function
    return integral_kde(kde_s, density_function=density_function)
def tau_s_t(points, lower_bounds, upper_bounds, bins=10,sample_points = 10000, bandwidth = "scott"):
    """
    Calculate the tau_s_t value using kernel density estimation.

    Args:
        points (numpy.ndarray): Array of points with shape (n, m) where n is the number of points and m is the dimensionality.
        lower_bounds (list): Array of lower bounds for each dimension.
        upper_bounds (list): Array of upper bounds for each dimension.
        bins (int): Number of bins to divide the time range into.
        sample_points (int): Number of points to sample within each bin.
        bandwidth (float): Bandwidth parameter for kernel density estimation.

    Returns:
        float: The tau_s_t value.

    """

    t_lower = lower_bounds[-1]
    t_upper = upper_bounds[-1]

    d = (t_upper - t_lower) / bins

    # Fit kernel density estimation for t dimension
    kde_t = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_t.fit(points[:, -1].reshape(-1, 1))

    ts = []
    for t in range(bins):
        bin_start = t_lower + t * d
        bin_end = t_lower + (t + 1) * d

        # Select points within the current bin
        points_t = points[(bin_start < points[:, -1]) & (points[:, -1] < bin_end)]

        # Check if there are points in the bin
        # Todo: make this work when there is 0 elements in the bin
        if len(points_t) == 0:
            # Skip empty bin
            continue

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
        t_val = np.exp(kde_s_t.score_samples([[0] * (points.shape[1] - 1)])) * p

        ts.append(t_val)

    return np.sum(ts) * d

