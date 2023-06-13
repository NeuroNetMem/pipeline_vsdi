"""
This module provides functions for performing dimensionality reduction on VSDI (Voltage-Sensitive Dye Imaging) data using PCA (Principal Component Analysis) and ICA (Independent Component Analysis).

Dependencies:
- numpy
- sklearn

Usage:
- Use the `pca` function to perform PCA on VSDI data and obtain the principal components and timecourses.
- Use the `pca_ica` function to perform PCA and ICA on VSDI data and obtain the fingerprint patterns, timecourses, and explained variance ratios.
"""

# Dimensionality reduction
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def pca(vsdi, raw_mask=None, n_comp=10, normalize=True):
    """
    Perform Principal Component Analysis (PCA) on vsdi data.

    The function reshapes the vsdi data, optionally applies a raw mask, and then performs PCA on the reshaped data.
    It returns the principal components (also known as "fingerprints") and their corresponding timecourses.

    Parameters
    ----------
    vsdi : ndarray
        The input vsdi data to be analyzed. It should be a 3D numpy array with dimensions (height, width, time).
    
    raw_mask : ndarray, optional
        A mask to be applied to the vsdi data. The mask should be a 2D numpy array with the same spatial dimensions 
        as the vsdi data. Pixels outside the mask are set to zero. If None, no mask is applied. Default is None.

    n_comp : int, optional
        The number of principal components to compute. Default is 10.

    normalize : bool, optional
        Whether to normalize the input data before performing PCA. If True, the data are normalized to have zero mean 
        and unit variance. Default is True.

    Returns
    -------
    fingerprints : ndarray
        The principal components of the vsdi data. This is a 2D numpy array where each row corresponds to a principal 
        component and each column corresponds to a pixel. The components are ordered by their explained variance, with 
        the component that explains the most variance first.

    timecourses : ndarray
        The timecourses corresponding to each principal component. This is a 2D numpy array where each row corresponds 
        to a principal component and each column corresponds to a time point.

    Examples
    --------
    >>> vsdi_data = np.random.rand(100, 100, 1000)  # Mock vsdi data
    >>> fingerprints, timecourses = pca(vsdi_data)
    """
    # reshape vsdi data to (time, pixels)
    X = vsdi.transpose(2, 0, 1)
    # store original shape
    # vsdi_shape = vsdi.shape
    if raw_mask is not None:
        # put out-of-mask values to zero
        X = X[:, raw_mask]
    else:
        # flatten 2d pixels to array
        X = X.reshape(vsdi.shape[0], vsdi.shape[1]*vsdi.shape[2])

    # Create a pipeline with PCA
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10))
    ])

    # Fit the pipeline to vsdi data
    out = pipe.fit(X)
    fingerprints = out.named_steps['pca'].components_
    timecourses = fingerprints @ X.T

    return fingerprints, timecourses


def pca_ica(vsdi, mask):
    """
    Perform PCA and ICA on VSDI data.

    Parameters
    ----------
    vsdi : ndarray
        Array containing VSDI data.
    mask : ndarray
        Array representing the mask to be applied to the VSDI data.

    Returns
    -------
    fingerprints : ndarray
        Array containing the fingerprint patterns obtained from ICA.
    timecourses : ndarray
        Array containing the timecourses obtained from ICA.
    explained_variance : ndarray
        Array containing the explained variance ratios from PCA.
    """
    X = vsdi.transpose(2, 0, 1)
    X = X*mask
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

    # Create a pipeline with PCA and ICA
    pipe = Pipeline([
        ('pca', PCA(n_components=10)),
        ('ica', FastICA(n_components=10, max_iter=200,
                        random_state=1, whiten='unit-variance'))
    ])

    out = pipe.fit(X)

    fingerprints = out.named_steps["ica"].components_ @ out.named_steps["pca"].components_
    timecourses = fingerprints @ X.T

    explained_variance = out.named_steps["pca"].explained_variance_ratio_

    return fingerprints, timecourses, explained_variance
