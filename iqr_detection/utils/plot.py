
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def display_multiple_df(data_list, outliers_data_list=None):
    """
    Display multiple numerical 1-D data set with a kernel density.
    Outliers can be added to be displayed.
    This function is mainly use as a data science demonstration tools.

    :param data_list: [(pd.DataFrame), (pd.DataFrame), ...]
        List of pd.DataFrame - numerical 1-D data
    :param outliers_data_list: [(pd.DataFrame), (pd.DataFrame), ...]
        List of pd.DataFrame - numerical 1-D data which represent outliers.
        Default is None.
    :return: Nothing (only display)

    Example:
    --------
    >>> from iqr_detection.utils.generate_outliers import generate_outliers_1d
    >>>
    >>> data_gauss_ext = generate_outliers_1d(how="gaussian+extreme", output_format="pandas")
    >>> display_multiple_df([data_gauss_ext, data_gauss_ext, data_gauss_ext])
    '''
    """

    # Plot style
    nb_cols = len(data_list)
    font = {'weight': 'medium', 'size': 22}
    plt.rc('font', **font)
    fig, ax = plt.subplots(nrows=2, ncols=nb_cols, figsize=(23, 10))
    fig.suptitle('Data display with kernel density below (x: index ; y: value)')

    # Plot architecture
    for idx in range(len(data_list)):
        data = data_list[idx]
        ax[0, idx].plot(data)

        # Plot KDE curve
        kde_curve, x_plot = _generate_kde(data)
        ax[1, idx].plot(x_plot, kde_curve, linestyle='-', color='darkorange')
        ax[1, idx].fill(x_plot, kde_curve, fc='#f2e7da')

        # Plot data at the bottom
        ax[1, idx].plot(data, -0.005 - 0.003 * np.random.random(data.shape[0]), '+k')

        # Plot outliers if needed
        if outliers_data_list is not None:
            outliers_data = outliers_data_list[idx]
            ax[0, idx].plot(outliers_data, marker='o', linestyle="None")
            ax[1, idx].plot(outliers_data, -0.005 - 0.003 * np.random.random(outliers_data.shape[0]),
                            linestyle="None", marker='o')
    plt.show()


def display_one_df(data, outliers_data=None):
    """
    Display numerical 1-D data set with a kernel density.
    Outliers can be added to be displayed.
    This function is mainly use as a data science demonstration tools.

    :param data: (pd.DataFrame) Numerical 1-D data.
    :param outliers_data: (pd.DataFrame) Numerical 1-D data which represent outliers.
        Default is None.
    :return: Nothing (only display)

    Example:
    --------
    >>> from iqr_detection.utils.generate_outliers import generate_outliers_1d
    >>>
    >>> data_gauss_ext = generate_outliers_1d(how="gaussian+extreme", output_format="pandas")
    >>> display_multiple_df(data_gauss_ext)
    '''
    """

    # Plot style
    font = {'weight': 'medium', 'size': 22}
    plt.rc('font', **font)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(23, 10))
    fig.suptitle('Data display with kernel density below (x: index ; y: value)')

    # Plot architecture
    ax[0].plot(data)
    kde_curve, x_plot = _generate_kde(data)

    # Plot KDE curve
    ax[1].plot(x_plot, kde_curve, linestyle='-', color='darkorange')
    ax[1].fill(x_plot, kde_curve, fc='#f2e7da')

    # Plot data at the bottom
    ax[1].plot(data, -0.005 - 0.003 * np.random.random(data.shape[0]), '+k')

    # Plot outliers if needed
    if outliers_data is not None:
        ax[0].plot(outliers_data, marker='o', linestyle="None")
        ax[1].plot(outliers_data, -0.005 - 0.003 * np.random.random(outliers_data.shape[0]),
                   linestyle="None", marker='o')

    plt.show()


def _generate_kde(_data):
    """
    Generate a Kernel Density from a 1-D numerical data set.

    :param _data: (pd.DataFrame) numerical 1-D data set.
    :return: y, x values for the Kernel Density.
    """

    # Fit the Kernel Density with the data
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(_data)

    # Generate point for kernel density
    _min_data = np.amin(_data) - np.abs(np.amax(_data)) * 0.1
    _max_data = np.amax(_data) + np.abs(np.amax(_data)) * 0.1
    _x_plot = np.linspace(_min_data, _max_data, 500)
    log_dens = kde.score_samples(_x_plot)
    _kde_curve = np.exp(log_dens)
    return _kde_curve, _x_plot
