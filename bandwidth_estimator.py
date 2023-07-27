import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate


def scott_bandwidth(data):
    """
    Calculate the bandwidth for Kernel Density Estimation using Scott's rule of thumb.

    Parameters:
        data (array-like): Input data array.

    Returns:
        float: Bandwidth value.
    """
    n = len(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    h_scott = 1.06 * std_dev * (n ** (-0.2))
    return h_scott

def silverman_bandwidth(data):
    """
    Calculate the bandwidth for Kernel Density Estimation using Silverman's rule of thumb.

    Parameters:
        data (array-like): Input data array.

    Returns:
        float: Bandwidth value.
    """
    n = len(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    iqr = np.subtract(*np.percentile(data, [75, 25]))  # Interquartile range (IQR)
    h_silverman = 0.9 * min(std_dev, iqr / 1.34) * (n ** (-0.2))
    return h_silverman


def improved_sheather_jones_bandwidth(data, var_type='c'):
    """
    Calculate the bandwidth for Kernel Density Estimation using the Improved Sheather Jones method.

    Parameters:
        data (array-like): Input data array.
        var_type (str): String indicating the variable type.
                        'c': Continuous variable
                        'u': Unordered (discrete) variable
                        'o': Ordered (discrete) variable

    Returns:
        array-like: Bandwidth values for each dimension of the input data.
    """
    var_type = var_type * data.shape[1]
    kde = KDEMultivariate(data, var_type=var_type)
    return kde.bw[0]

def get_bandwidth(points, bandwidth_type):
    """
    Get the bandwidth value for Kernel Density Estimation based on the specified type.

    Parameters:
        points (array-like): Input data points.
        bandwidth_type (str or float): Bandwidth type or a numeric value.

    Returns:
        float: Bandwidth value.
    """

    bandwidth = None

    if bandwidth_type == "scott":
        bandwidth = scott_bandwidth(points)

    elif bandwidth_type == "silverman":
        bandwidth = silverman_bandwidth(points)

    elif bandwidth_type == "ISJ":
        bandwidth = improved_sheather_jones_bandwidth(points)

    elif isinstance(bandwidth_type, (int, float)):
        bandwidth = bandwidth_type

    if bandwidth is None or bandwidth == 0:
        bandwidth = 1

    return bandwidth