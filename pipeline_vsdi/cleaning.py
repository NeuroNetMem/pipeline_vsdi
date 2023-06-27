import numpy as np


def normalize_vsdi(vsdi, mask):
    # Transpose to this shape (time, x, y)
    v_t = vsdi.transpose(2, 0, 1)
    # Mask vsdi
    v_masked = v_t * mask
    v_reshaped = v_masked.reshape(v_masked.shape[0], v_masked.shape[1]*v_masked.shape[2])
    # Normalize usign Standard Scaler
    v_norm = StandardScaler().fit_transform(v_reshaped)
    v_norm_reshaped = v_norm.reshape(v_masked.shape[0], v_masked.shape[1], v_masked.shape[2])
    # Transpose to this shape (x, y, time)
    v_norm_original = v_norm_reshaped.transpose(1, 2, 0)
    
    return v_norm_original

def find_outliers(vsdi, nsigma=3):
    """
    Identify the start and end indices of outlier frames in VSDI data

    Parameters
    ----------
    vsdi : numpy ndarray
        vsdi data in format (h, w, time)
    nsigma : float
        Number of standard deviations to consider an outlier

    Returns
    -------
    outliers_subsets : list of tuples
        Each tuple contains the start and end indices of an outlier subset
    """
    mean_vsdi = np.mean(vsdi, axis=(0, 1))
    # How many standard deviations away a value is from the mean
    zscore = stats.zscore(mean_vsdi, axis=0, ddof=0, nan_policy='propagate')
    # zscore = StandardScaler().fit_transform(vsdi)
    
    # Get index of outliers from vsdi presenting average activity higher than 4 sigma
    outliers = np.argwhere((zscore > nsigma) | (zscore < -nsigma)).ravel()

    subsets = []
    start = 0
    end = 0

    # Iterate over array
    while end < len(outliers):
        # Get first and last frame of each subset
        while end + 1 < len(outliers) and outliers[end + 1] - outliers[start] == end - start + 1:
            end += 1
        # Save first and last frame of each outlier subset
        subsets.append((outliers[start], outliers[end]))
        start = end = end + 1

    return np.array(subsets)


def clean_outliers(vsdi, nsigma=3):
    """
    Correct outliers by pixel-wise linear interpolation in VSDI data

    Parameters
    ----------
    vsdi : numpy ndarray
        vsdi data in format (h, w, time)
    nsigma : float
        Number of standard deviations to consider an outlier

    Returns
    -------
    vsdi : numpy ndarray
        vsdi data with outliers corrected
    """
    vsdi = vsdi.copy()
    # Get first and last frame of each subset
    outliers_subsets = find_outliers(vsdi, nsigma)

    # Set outlier frames to the mean between the previous and next frame
    for i in range(len(outliers_subsets)):
        start = outliers_subsets[i][0]
        end = outliers_subsets[i][1]
        if start == 0:
            vsdi[:, :, start:end+1] = np.tile(vsdi[:, :, end+1][:, :, np.newaxis], (1, 1, end - start + 1))
        elif end == vsdi.shape[2]-1:
            vsdi[:, :, start:end+1] = np.tile(vsdi[:, :, start-1][:, :, np.newaxis], (1, 1, end - start + 1))
        else:
            average = np.divide(np.add(vsdi[:, :, (start-1)][:, :, np.newaxis], vsdi[:, :, (end+1)][:, :, np.newaxis]), 2)
            vsdi[:, :, start:end+1] = average
    return vsdi