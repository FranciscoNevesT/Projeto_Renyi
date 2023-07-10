import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def integral_kde(kde, lower_bounds, upper_bounds, num_points=10, density_function = lambda x : x):
    """
    Calculate the integral value of a kernel density estimation (KDE) function.

    Args:
        kde (sklearn.neighbors.KernelDensity): The fitted KDE object.
        lower_bounds (list): List of lower bounds for each dimension.
        upper_bounds (list): List of upper bounds for each dimension.
        num_points (int): Number of points to generate along each dimension for evaluating the KDE.
        density_function (function): Function to apply to the density values before integration.

    Returns:
        float: The integral value.

    """
    #TODO: Implement a more precise way to calc integral

    # Generate points on the axis for evaluating the PDF
    points = [np.linspace(lower, upper, num_points) for lower, upper in zip(lower_bounds, upper_bounds)]
    meshgrid = np.meshgrid(*points)
    samples = np.column_stack([axis.ravel() for axis in meshgrid])

    # Compute the estimated density values
    log_density = kde.score_samples(samples)
    density = np.exp(log_density)
    density = density_function(density)

    # Compute the integral value
    volume = np.prod([(upper - lower) / num_points for lower, upper in zip(lower_bounds, upper_bounds)])
    integral_value = np.sum(density) * volume

    return integral_value

def tau_s(points, lower_bounds, upper_bounds,bandwidth=0.05):
    """
    Calculate the tau_s value using kernel density estimation.

    Args:
        points (numpy.ndarray): Array of points with shape (n, m) where n is the number of points and m is the dimensionality.
        lower_bounds (list): List of lower bounds for each dimension.
        upper_bounds (list): List of upper bounds for each dimension.
        bandwidth (float): Bandwidth parameter for kernel density estimation.

    Returns:
        float: The tau_s value.

    """
    # Fit kernel density estimation for s dimension
    kde_s = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_s.fit(points[:, :-1].reshape(-1, points.shape[1] - 1))

    # Define density function for integration
    density_function = lambda x: x ** 2

    # Calculate tau_s using integral_kde function
    return integral_kde(kde_s, lower_bounds=lower_bounds, upper_bounds=upper_bounds, num_points=100,
                        density_function=density_function)

def tau_s_t(points, lower_bounds, upper_bounds, bins=10,sample_points = 1000, bandwidth = 0.05):
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

        # Calculate p using kernel density estimation for t dimension
        p = np.exp(kde_t.score_samples([[bin_start]]))[0]

        # Calculate t_val by evaluating kernel density estimation for s-t dimension at 0 and multiplying by p
        t_val = np.exp(kde_s_t.score_samples([[0] * (points.shape[1] - 1)])) * p

        ts.append(t_val)

    return np.sum(ts) * d

