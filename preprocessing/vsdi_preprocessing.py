##########################################################
######################## PACKAGES ########################
##########################################################
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import scipy.stats as stats
import statsmodels.api as sm


##########################################################

def peak_distance(component, smooth_window_size, hist_nbins):
    # Create a histogram
    n, bins = np.histogram(component, bins=hist_nbins)
    # Define the window size for the moving average
    window_size = smooth_window_size
    # Create the moving average kernel
    kernel = np.ones(window_size) / window_size
    # Convolve the histogram data with the moving average kernel
    smoothed = np.convolve(n, kernel, mode='same')
    bins_adjusted = bins[:-1]
    # Find the x-axis location of the highest point in the histogram
    x_max = bins_adjusted[np.argmax(smoothed)]

    return x_max


def check_bimodal(components, th=1.0, smooth_window_size=5, hist_nbins=1000):
    """
    Test for bimodality using moving average to smooth the histogram
    and get the x-axis location of the highest point in the histogram

    Parameters
    ----------
    components : numpy ndarray (components, time)
        Timecourse of a single component
    threshold : float
        Threshold for bimodal distribution identification (distance from peak to zero)

    Returns
    -------
    bimodal_components : list
        List of components that are bimodal (1 if bimodal, 0 if not)
    """
    # Initialize bimodal_components list
    bimodal_components = np.zeros(components.shape[0])
    for i in range(components.shape[0]):
        # Get peak distance
        x_max = peak_distance(components[i], smooth_window_size, hist_nbins)
        # Check if peak distance is above threshold
        if abs(x_max) > th:
            bimodal_components[i] = 1

    return bimodal_components


def clean_hemodynamic_pca(vsdi, n_components=50, bimodal_th=1,
                          smooth_window=5, hist_nbins=1000, verbose=False):
    """
    Clean hemodynamic noise from vsdi data

    Parameters
    ----------
    vsdi : numpy array
        vsdi data in format (time, pixels_y, pixels_X)
    n_components : int
        Number of components to keep after PCA
    bimodal_th : float
        Threshold for bimodal distribution identification
    smooth_window : int
        Window size for smoothing the histogram of timecourse distribution
    hist_nbins : int
        Number of bins for histogram of timecourse distribution

    Returns
    -------
    clean_vsdi : numpy array in the same format as vsdi input
        vsdi data with hemodynamic noise removed
    """
    # reshape vsdi data to (time, pixels)
    vsdi = vsdi.transpose(2, 0, 1)
    # store original shape
    vsdi_shape = vsdi.shape
    # flatten 2d pixels to array
    vsdi = vsdi.reshape(vsdi.shape[0], vsdi.shape[1]*vsdi.shape[2])

    # put out-of-mask values to zero

    # run PCA
    pca = PCA(n_components=n_components)
    # fit PCA to vsdi data and project it to the new space
    timecourses = pca.fit_transform(vsdi)

    # get bimodal components indexes
    bad_components = check_bimodal(timecourses.T, th=bimodal_th,
                                   smooth_window_size=smooth_window,
                                   hist_nbins=hist_nbins)

    if verbose:
        with open('log.txt', 'a') as f:
            # Print explained variance ratio
            f.write(
                f'{sum(pca.explained_variance_ratio_[np.where(bad_components)]):.4f} ')
            # Print number of components
            f.write(f'{sum(bad_components):.0f}\n')

    # get clean timecourses and components
    good_timecourses = timecourses[:, np.logical_not(bad_components)]
    good_components = pca.components_[np.logical_not(bad_components)]

    # reconstruct vsdi data
    clean_vsdi = good_timecourses @ good_components
    clean_vsdi = clean_vsdi.reshape(
        vsdi_shape[0], vsdi_shape[1], vsdi_shape[2])
    clean_vsdi = clean_vsdi.transpose(1, 2, 0)

    return clean_vsdi


def clean_hemodynamics(vsdi, n_components=50, bimodal_th=0.8, ica_max_iter=200,
                       smooth_window=5,  hist_nbins=1000, verbose=False):
    """
    Clean hemodynamic noise from vsdi data

    Parameters
    ----------
    vsdi : numpy array
        vsdi data in format (time, pixels_y, pixels_X)
    n_components : int
        Number of components to keep after PCA
    bimodal_th : float
        Threshold for bimodal distribution identification
    smooth_window : int
        Window size for smoothing the histogram of timecourse distribution
    hist_nbins : int
        Number of bins for histogram of timecourse distribution

    Returns
    -------
    clean_vsdi : numpy array in the same format as vsdi input
        vsdi data with hemodynamic noise removed
    """
    # reshape vsdi data to (time, pixels)
    X = vsdi.transpose(2, 0, 1)
    # store original shape
    vsdi_shape = vsdi.shape
    # flatten 2d pixels to array
    X = X.reshape(vsdi.shape[0], vsdi.shape[1]*vsdi.shape[2])

    # Create a pipeline with PCA and ICA
    pipe = Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('ica', FastICA(n_components=n_components, max_iter=ica_max_iter,
                        random_state=88, whiten='unit-variance'))
    ])

    # Fit the pipeline to vsdi data
    timecourses = pipe.fit_transform(X)

    # get bimodal components indexes
    bad_components = check_bimodal(timecourses.T, th=bimodal_th,
                                   smooth_window_size=smooth_window,
                                   hist_nbins=hist_nbins)

    if verbose:
        with open('log.txt', 'a') as f:
            # Print number of components
            f.write(f'{sum(bad_components):.0f}\n')

    # get clean timecourses and components
    good_timecourses = timecourses[:, np.logical_not(bad_components)]
    good_components = pipe.named_steps['ica'].components_[
        np.logical_not(bad_components)]

    # reconstruct vsdi data
    clean_vsdi = good_components @ pipe.named_steps['pca'].components_
    clean_vsdi = good_timecourses @ clean_vsdi
    clean_vsdi = clean_vsdi.reshape(
        vsdi_shape[0], vsdi_shape[1], vsdi_shape[2])

    return clean_vsdi


def find_outliers_old(vsdi, mean_vsdi, std_vsdi, nsigma=4):
    # Get index of outliers from vsdi presenting average activity higher than 4 sigma
    outliers = np.argwhere((mean_vsdi > nsigma*std_vsdi)
                           | (mean_vsdi < -nsigma*std_vsdi)).ravel()

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


def find_outliers(vsdi, nsigma=6):
    mean_vsdi = np.mean(vsdi, axis=(0, 1))
    # How many standard deviations away a value is from the mean
    zscore = stats.zscore(mean_vsdi, axis=0, ddof=0, nan_policy='propagate')
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


def clean_outliers(vsdi, nsigma=6):
    """
    Correct outliers in VSDI data

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
            vsdi[:, :, start:end+1] = np.tile(vsdi[:, :, end+1]
                                              [:, :, np.newaxis], (1, 1, end - start + 1))
        elif end == vsdi.shape[2]-1:
            vsdi[:, :, start:end+1] = np.tile(vsdi[:, :, start-1]
                                              [:, :, np.newaxis], (1, 1, end - start + 1))
        else:
            average = np.divide(np.add(
                vsdi[:, :, (start-1)][:, :, np.newaxis], vsdi[:, :, (end+1)][:, :, np.newaxis]), 2)
            vsdi[:, :, start:end+1] = average
    return vsdi


def design_matrix(b_data):
    # Design matrix
    # Returns a matrix of size (time, 6) with the following columns:
    # 0: CS+ (2s)
    # 1: CS+ trace (1s)
    # 2: CS- (2s)
    # 3: CS- trace (1s)
    # 4: Reward (1s)
    # 5: Lick

    fps = 50

    Lick = b_data['Lick']
    CSp = b_data['CSp']
    CSn = b_data['CSn']
    frames = b_data['frames']

    length = len(frames)
    X = np.zeros((length, 6))

    # Iterate over lick events
    for i in range(len(Lick)):
        frame = np.argmin(np.abs(frames - Lick[i]))
        X[frame:frame+4, 5] = 1

    # Iterate over CSp events
    for i in range(len(CSp)):
        frame = np.argmin(np.abs(frames - CSp[i]))
        X[frame:frame + (fps * 2), 0] = 1
        frame += (fps * 2) + 1
        X[frame:frame + (fps * 1), 1] = 1
        frame += (fps * 1) + 1
        X[frame:frame + (fps * 1), 4] = 1

    # Iterate over CSn events
    for i in range(len(CSn)):
        frame = np.argmin(np.abs(frames - CSn[i]))
        X[frame:frame + (fps * 2), 2] = 1
        frame += (fps * 2) + 1
        X[frame:frame + (fps * 1), 3] = 1

    return X


def pca(vsdi, raw_mask=None, n_comp=10, normalize=True):
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


def pca_ica(vsdi, raw_mask, n_comp=10,z_score = True,ica_random_state=42,ica_max_iter=200):
    # reshape vsdi data to (time, pixels)
    X = vsdi.transpose(2, 0, 1)
    X = X*raw_mask
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    
    if z_score:
        X = StandardScaler().fit_transform(X)
    

    # Create a pipeline with PCA and ICA
    pipe = Pipeline([('pca', PCA(n_components=n_comp)),
        ('ica', FastICA(n_components=n_comp, max_iter=ica_max_iter,
                        random_state=ica_random_state, whiten='unit-variance'))
    ])

    out = pipe.fit(X)
    fingerprints = out.named_steps["ica"].components_ @ out.named_steps["pca"].components_
    timecourses = fingerprints @ X.T

    return fingerprints.reshape(n_comp,raw_mask.shape[0],raw_mask.shape[1]), timecourses


def glm(Y, X):
    # Fit Gaussian GLMs
    X = sm.add_constant(X)
    model = sm.GLM(Y, X, family=sm.families.Gaussian())
    results = model.fit()

    return results


def merge_masks(masks, threshold=0.5):
    """
    Merge masks by voting majority / logical OR 

    Input: masks (n_masks, h, w)
    Output: mask_mean (h, w)
    """
    mask_mean = np.mean(masks, axis=0)
    mask_mean[mask_mean > threshold] = 1
    mask_mean[mask_mean <= threshold] = 0
    return mask_mean
