
import numpy as np
import pandas as pd
import warnings


def generate_outliers_1d(how="gaussian", output_format="pandas", seed=42):
    """
    Generate a 1 Dimension dataset. Data are random, but in a fixed range.
    This function is mainly uses as data science demonstration tools

    :param how: (str) Can be "gaussian", "extreme", or "gaussian+extreme".
        Default is "gaussian".
    :param output_format: (str) Can be "numpy" or "pandas".
        Default is "pandas"
    :param seed: (int) Random seed.
        Default is 42.

    :return: np.array or pd.DataFrame according to output_format selected.
    """

    def _gaussian():
        # Generate random gaussian value between 10 and 30
        random = 20 + np.random.randn(200) * 10
        # Add noise
        return random + np.random.rand(200) * 2

    def _extreme():
        # Generate random gaussian value between 25 and 50
        random = (np.random.rand(190) + 1) * 25
        # Generate random gaussian value between 5 and 10
        random_low = (np.random.rand(5) + 1) * 5
        # Generate random gaussian value between 50 and 100
        random_high = (np.random.rand(5) + 1) * 50
        # Concatenate all together
        _output = np.concatenate((random, random_low, random_high), axis=None)
        # return shuffled data
        np.random.shuffle(_output)
        return _output

    def _gaussian_extreme():
        # Generate random gaussian value between 20 and 40
        random = 30 + np.random.randn(190) * 10
        # Generate random gaussian value between 0 and 1
        random_low = 0.5 + np.random.randn(5) * 0.5
        # Generate random gaussian value between 55 and 65
        random_high = 60 + np.random.randn(5) * 5
        # Concatenate all together
        _output = np.concatenate((random, random_low, random_high), axis=None)
        # Shuffle data
        np.random.shuffle(_output)
        return _output

    np.random.seed(seed)
    if how == "gaussian":
        output = _gaussian()
    elif how == "extreme":
        output = _extreme()
    elif how == "gaussian+extreme":
        output = _gaussian_extreme()
    else:
        output = _gaussian()
        warnings.warn("Wrong input for \'how\' parameter, gaussian method is used.")

    if output_format == "numpy":
        return output
    elif output_format == "pandas":
        return pd.DataFrame(data=output, columns=["values"])
    else:
        warnings.warn("Wrong input for \'output_format\' parameter, pd.DataFrame is used.")
        return output
