
from iqr_detection.utils.plot import display_multiple_df
from iqr_detection.utils.generate_outliers import generate_outliers_1d

# Generate 3 outliers dataset
data_gaussian = generate_outliers_1d(how="gaussian", output_format="pandas")
data_extreme = generate_outliers_1d(how="extreme", output_format="pandas")
data_gauss_ext = generate_outliers_1d(how="gaussian+extreme", output_format="pandas")

# Display generated dataset
display_multiple_df([data_gaussian, data_extreme, data_gauss_ext])


def iqr_detection(df_x, k=1):
    """
    Process IQR outlier detection. The input must be an one dimensional feature space.
    To adjust IQR detection, please change 'k' (k must always be >=0).

    :param df_x: (pd.Series or pd.DataFrame)
    :param k: (int) K parameter in the IQR outlier detection
    :return: (pd.Series) Boolean series.
        True: The value is an outlier
        False: The value is not an outlier
    """
    q1 = df_x.quantile(0.25).values[0]
    q3 = df_x.quantile(0.75).values[0]
    iqr = q3 - q1
    return df_x.apply(lambda x: (x > q3+k*iqr) | (x < q1-k*iqr))


# Apply IQR detection
data_gaussian["is_outlier"] = iqr_detection(data_gaussian)
data_extreme["is_outlier"] = iqr_detection(data_extreme)
data_gauss_ext["is_outlier"] = iqr_detection(data_gauss_ext)

# Data without outliers
gaussian_without_outliers = data_gaussian[~data_gaussian["is_outlier"]].drop(columns="is_outlier")
extreme_without_outliers = data_extreme[~data_extreme["is_outlier"]].drop(columns="is_outlier")
gauss_ext_without_outliers = data_gauss_ext[~data_gauss_ext["is_outlier"]].drop(columns="is_outlier")

# Keep only outliers
gaussian_outliers = data_gaussian[data_gaussian["is_outlier"]].drop(columns="is_outlier")
extreme_outliers = data_extreme[data_extreme["is_outlier"]].drop(columns="is_outlier")
gauss_ext_outliers = data_gauss_ext[data_gauss_ext["is_outlier"]].drop(columns="is_outlier")

# Display result
display_multiple_df([gaussian_without_outliers, extreme_without_outliers, gauss_ext_without_outliers],
                    outliers_data_list=[gaussian_outliers, extreme_outliers, gauss_ext_outliers])
