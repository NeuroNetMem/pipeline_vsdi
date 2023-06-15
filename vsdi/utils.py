# Import data
from pathlib import Path
from scipy.io import loadmat

# Logging
import logging
from datetime import datetime
import time
from tqdm import tqdm

# Data management
import numpy as np
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import pickle

# Dimensionality reduction
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline

# VSDI components resizing
import cv2
from functools import reduce

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from statannotations.Annotator import Annotator

# Decoder
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import warnings
from sklearn.exceptions import ConvergenceWarning

################
### BEHAVIOR ###
################

def make_design_matrix(b_data, event_sequences, fps=50):
    """
    Generates a design matrix based on the provided event sequences.

    The design matrix contains rows corresponding to frames and columns corresponding to events. Each column index in the output matrix represents an event in the event sequences. The number of columns in the design matrix equals the maximum column index specified in the event sequences plus one.

    Parameters
    ----------
    event_sequences : dict
        A dictionary where keys are event names and values are lists of tuples. Each tuple contains the name of an event, the duration of the event in seconds, and the column index in the output matrix.

    Returns
    -------
    matrix : ndarray
        Design matrix where each row corresponds to a frame and each column corresponds to an event.

    Examples
    --------
    >>> event_sequences = {
    ...     'Lick': [('Lick', 1, 5)],
    ...     'CSp': [('CS+', 2, 0), ('CS+ Trace', 1, 1), ('Reward', 1, 4)],
    ...     'CSn': [('CS-', 2, 2), ('CS- Trace', 1, 3)],
    ... }
    >>> design_matrix(event_sequences)
    # Output: 
    # matrix with columns: 
    # 0: CS+ (2s)
    # 1: CS+ trace (1s)
    # 2: CS- (2s)
    # 3: CS- trace (1s)
    # 4: Reward (1s)
    # 5: Lick
    """
    frames = b_data['frames']
    length = len(frames)
    n_columns = max(col for seq in event_sequences.values() for _, _, col in seq) + 1
    X = np.zeros((length, n_columns))

    for event, sequence in event_sequences.items():
        event_data = b_data[event]
        
        for i in range(len(event_data)):

            frame = np.argmin(np.abs(frames - event_data[i]))
            
            for event_name, duration, column in sequence:
                duration_unit = fps * duration if fps * duration != 0 else 1 # extreme case for licks
                X[frame:(frame + int(duration_unit)), column] = 1
                frame += fps * duration
                
    return X

def subsets(Set):
    """
    Finds the starting indices of subsets of consecutive numbers within the provided set.

    This function identifies series of consecutive numbers (subsets) within the input set.
    It returns a numpy array of the starting indices of each identified subset.

    Parameters
    ----------
    Set : list or array-like
        List or array-like object containing the series of numbers to be partitioned into subsets.

    Returns
    -------
    subsets : ndarray
        A numpy array containing the starting indices of each identified subset of consecutive numbers.

    Examples
    --------
    >>> subsets([1, 2, 3, 5, 6, 8, 9, 10])
    # Output: array([1, 5, 8])

    Notes
    -----
    The subsets are defined as series of consecutive numbers. For example, in the array
    [1, 2, 3, 5, 6, 8, 9, 10], the subsets of consecutive numbers are [1, 2, 3], [5, 6] 
    and [8, 9, 10], so the function returns the starting points [1, 5, 8].
    """
    subsets = []
    start = 0
    end = 0
    
    # Iterate over array
    while end < len(Set):
        
        # Get first and last frame of each subset
        while end + 1 < len(Set) and Set[end + 1] - Set[start] == end - start + 1:
            end += 1
        
        # Save first and last frame of each outlier subset
        subsets.append(Set[start])
        start = end = end + 1
    
    return np.array(subsets)

def get_trials(feature):
    """
    Finds the starting indices of trials within the provided design matrix vector.

    This function identifies the start of each trial within the input design matrix vector,
    where trials are defined as sequences where the feature equals 1.

    Parameters
    ----------
    feature : list or array-like
        List or array-like object representing a design matrix vector. The start of each 
        trial is defined as the index where the feature value equals 1.

    Returns
    -------
    feature_trials : ndarray
        A numpy array containing the starting indices of each identified trial.

    Examples
    --------
    >>> get_trials([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    # Output: array([2, 7])

    Notes
    -----
    The trials are defined as sequences where the feature equals 1. For example, in the array
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0], the trials are [1, 1, 1] and [1, 1], so the function 
    returns the starting points [2, 7].
    """
    # Find the indices where feature is 1
    feature_trials = subsets(np.where(feature == 1)[0])
    
    return feature_trials

def compute_lick_rate(data, trial_onsets, fps):
    """
    Compute the lick rate for each trial.

    For each trial, the function computes the number of licks during a 'baseline' period 
    of 1 second before CS presentation and during a 'trace' period of 1 second starting 
    2 seconds after CS presentation. The lick rate for the trial is then computed as 
    the number of licks in the trace period minus the number of licks in the baseline period.

    Parameters
    ----------
    data : ndarray
        1D numpy array containing lick data. Each element corresponds to a frame, and its 
        value indicates whether a lick occurred in that frame (1) or not (0).
        
    trial_onsets : ndarray
        1D numpy array containing the onset times (in frames) of each trial.
        
    fps : int
        Frame rate of the data (frames per second).
        
    Returns
    -------
    lick_rates : list
        List of lick rates for each trial.

    Examples
    --------
    >>> lick_data = np.random.randint(0, 2, 600)  # Mock lick data
    >>> trial_onsets = np.array([100, 300, 500])  # Mock trial onset times
    >>> fps = 30  # Frame rate
    >>> lick_rates = compute_lick_rate(lick_data, trial_onsets, fps)
    """
    lick_rates = []
    
    for i in range(len(trial_onsets)):
        
        trial_start = trial_onsets[i]
        # Baseline -1 s before CS presentation
        baseline_licks = np.sum(data[(trial_start-fps):trial_start]) 
        # Number of licks in trace period
        trace_licks = np.sum(data[trial_start+(2*fps):trial_start+(3*fps)])
        # Corrected lick rate
        lick_rate = trace_licks - baseline_licks
        
        lick_rates.append(lick_rate)
    
    return lick_rates

def store_indices(animals, days, sessions, outpath):
    """
    Store cumulative index (based on X matrix length) for each session, day, and animal,
    as well as trial indices for CS+ and CS- trials in a nested dictionary structure.

    :param animals: List of animal names
    :param days: List of days
    :param sessions: List of sessions
    :param outpath: Path to the output directory
    :return: A nested dictionary containing the cumulative indices and trial indices
    """
    # Initialize an empty dictionary
    X = {animal: {day: {} for day in days} for animal in animals}

    # Iterate over all animals, days, and sessions
    for animal in animals:
        for day in days:
            # Initialize the cumulative count
            cumulative_count = 0

            for session in sessions:
                # Load design matrix
                X_matrix = np.loadtxt(outpath.joinpath(f"X_{animal}_{day}_{session}.csv"), delimiter=",")
                
                # Here we are assuming 'CS+' info is in column 0 and 'CS-' info is in column 2
                trial_types = ['CS+', 'CS-']
                for trial_type, column in zip(trial_types, [0, 2]):
                    feature = X_matrix[:, column]
                    trial_indices = get_trials(feature)
                    
                    # Store the trial indices in the nested dictionary
                    if session not in X[animal][day]:
                        X[animal][day][session] = {}
                    X[animal][day][session][trial_type] = trial_indices
                
                # Update the cumulative count
                cumulative_count += len(X_matrix)

                # Store the cumulative count in the nested dictionary
                X[animal][day][session]['Cumulative Index'] = cumulative_count

    return X


############
### VSDI ###
############

def pca_ica(vsdi, mask):
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

def find_bounding_rectangle(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return (x, y), (x + w, y + h)
    else:
        return None
    
def crop_to_bounds(image, bounds):
    return image[bounds[0][1]:bounds[1][1],bounds[0][0]:bounds[1][0]] 

def resize(mask_list):
    max_h = np.max([x.shape[0] for x in mask_list])
    max_w = np.max([x.shape[1] for x in mask_list])
    
    resized = []
    for m in mask_list:
        resized_mask = cv2.resize(m, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        resized.append(resized_mask)
    
    return resized

def resize_video(video_list):
    max_h = np.max([c.shape[0] for c in video_list])
    max_w = np.max([c.shape[1] for c in video_list])
    resized_vsdi = []

    for arr in video_list:
        # The last dimension (time) is not affected by resizing,
        # so we need to iterate over it
        reshaped_images = []
        for i in range(arr.shape[2]):
            # Resize each 2D slice along the third axis
            reshaped_img = cv2.resize(arr[:,:,i], (max_w, max_h), interpolation=cv2.INTER_CUBIC)
            reshaped_images.append(reshaped_img)

        # Stack the reshaped images along the third axis
        resized_vsdi.append(np.dstack(reshaped_images))
    
    return(resized_vsdi)

def filter_data2(data, onset, offset, idxs):
    fragments = []
    # For ids
    for idx in idxs:
        # Check data dimensionality
        if len(data.shape) == 1:
            # Append data slice
            fragment = data[idx+onset:idx+offset]
        else:
            slices = [slice(None)] * (data.ndim - 1)
            slices.append(slice(idx+onset, idx+offset))
            fragment = data[tuple(slices)]
        fragments.append(fragment)
    # Return ndarrray of fragments
    return np.stack(fragments)

def filter_data(data, onset, offset, idxs):
    """
    Get ndarray of data fragments based on index (idx+onset to idx+offset)
    """
    fragments = []
    for idx in idxs:
        if len(data.shape) == 1:
            # Manage extreme cases (when onset offset falls outside data) Why are there onsets at the end of the data?
            if idx + onset < 0 or idx + offset > data.shape[0]:
                start = max(idx + onset, 0)
                end = min(idx + offset, data.shape[0])
                length = offset - onset
                fragment = np.full(length, 0) # Fill missing fragment of data with value
                fragment[:end-start] = data[start:end]
            else:
                fragment = data[idx + onset:idx + offset]
        else:
            slices = [slice(None)] * (data.ndim - 1)
            if idx + onset < 0 or idx + offset > data.shape[-1]:
                start = max(idx + onset, 0)
                end = min(idx + offset, data.shape[-1])
                length = offset - onset
                fragment = np.full(data.shape[:-1] + (length,), 0) # Fill missing fragment of data with value
                slices_start_end = slices + [slice(start, end)]
                fragment[..., :end-start] = data[tuple(slices_start_end)]
            else:
                slices.append(slice(idx + onset, idx + offset))
                fragment = data[tuple(slices)]
        fragments.append(fragment)
    return np.stack(fragments)

def split_data(data, labels, axis=-1, attr=1):
    """
    Splits the provided data based on the given labels.

    The function partitions a multidimensional data array into smaller slices, 
    based on the shapes provided in the labels list. The slices are then stored in a dictionary, 
    where the keys are determined by the `attr` attribute of the info list.

    Parameters
    ----------
    data : ndarray
    	Multidimensional array to be partitioned.
    labels : list
    	List of tuples, where each tuple contains an info list and a shape integer. 
    	Example: 
    	[
    	 (['A07', 'Day1', 'ATC1'], 30000),
    	 (['A07', 'Day1', 'ATC2'], 30000)
    	]
    axis : int, optional
    	The axis along which the data array is to be sliced, by default -1 (the last axis).
    attr : int, optional
    	The index of the attribute in the info list that will be used as a key in the resulting dictionary, 
    	by default 1 (the second item).

    Returns
    -------
    partitioned_data : dict
    	A dictionary where keys are the extracted attribute from the info list, 
    	and values are the partitioned slices of the input data.

    Examples
    --------
    >>> data = np.arange(60000).reshape((2, 30000))
    >>> labels = [(['A07', 'Day1', 'ATC1'], 30000), (['A07', 'Day1', 'ATC2'], 30000)]
    >>> split_data(data, labels, axis=0, attr=1)
    {'Day1': [array([0, 1, ..., 29998, 29999]), array([30000, 30001, ..., 59998, 59999])]}
    """
    # Start index for slicing
    start = 0
    # Partitioned data dictionary
    partitioned_data = {}

    for info, shape in labels:
        day = info[attr]  # Extracting 'Day' attribute from info list
        
        # Slicing the array
        slices = [slice(None)] * data.ndim  # Create as many slices as dimensions
        slices[axis] = slice(start, start+shape)  # Modify the slice for the specified axis
        sliced = data[tuple(slices)]
        
        # Storing the sliced array into dictionary
        if day in partitioned_data:
            partitioned_data[day].append(sliced)
        else:
            partitioned_data[day] = [sliced]
        
        # Updating start index for next slice
        start += shape
        
    return partitioned_data

################
### DECODING ###
################

def train_SVM_day(t_csp_day, t_csn_day, time_bin_size, n_splits=5, n_shuffles=100, n_iter=1000, tol=1e-4):
    # concatenate the two ndarrays
    t_all_day = np.concatenate((t_csp_day, t_csn_day), axis=0)

    # Create time points based on time bin size
    time_points = range(0, t_all_day.shape[2], time_bin_size)

    # initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits)

    # prepare the label vector
    stim_type = np.array([0] * t_csp_day.shape[0] + [1] * t_csn_day.shape[0])

    avg_performance = []
    std_performance = []
    surrogate_performance = []

    # for each time bin, train a separate SVM classifier
    for i in time_points:
        # Average within time bin for SVM
        X_bin = np.mean(t_all_day[:, :, i:i+time_bin_size], axis=2)

        # flatten the data for SVM
        X = X_bin.reshape(t_all_day.shape[0], -1)
        y = stim_type

        accuracy_scores = []
        surrogate_scores = []
        
        for train_index, test_index in skf.split(X, y):
            print(f'Computing time bin {int(i/time_bin_size)}/{len(time_points)}')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train SVM
            clf = svm.LinearSVC(max_iter=n_iter, tol=tol)
            clf.fit(X_train, y_train)

            # calculate accuracy
            accuracy = f1_score(y_test, clf.predict(X_test))
            accuracy_scores.append(accuracy)

            # Surrogate analysis
            for _ in tqdm(range(n_shuffles)):
                y_train_shuffle = shuffle(y_train)
                clf.fit(X_train, y_train_shuffle)
                surrogate_accuracy = f1_score(y_test, clf.predict(X_test))
                surrogate_scores.append(surrogate_accuracy)

        # calculate average and standard deviation of accuracy for the current time point
        avg_performance.append(np.mean(accuracy_scores))
        std_performance.append(np.std(accuracy_scores))
        surrogate_performance.append(surrogate_scores)

    avg_performance = np.asarray(avg_performance)
    std_performance = np.asarray(std_performance)
    surrogate_performance = np.asarray(surrogate_performance)
    
    return avg_performance, std_performance, surrogate_performance

def train_SVM_baseline2(t_sound_day, t_baseline_day, time_bin_size, n_splits=5, n_shuffles=100, n_iter=1000, tol=1e-4):
    # Create time points based on time bin size
    time_points = range(0, t_sound_day.shape[2], time_bin_size)

    # initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits)

    avg_performance = []
    std_performance = []
    surrogate_performance = []

    # for each time bin, train a separate SVM classifier
    for i in time_points:
        print(f'Computing time bin {int(i/time_bin_size)}/{len(time_points)}')

        # get the data for this time bin
        t_sound_bin = t_sound_day[:, :, i:i+time_bin_size]
        t_baseline_bin = np.tile(t_baseline_day, (1, 1, t_sound_bin.shape[2] // t_baseline_day.shape[2]))

        # concatenate the sound and baseline ndarrays for this time bin
        t_all_bin = np.concatenate((t_sound_bin, t_baseline_bin), axis=0)

        # prepare the label vector
        y = np.array([0] * t_sound_bin.shape[0] + [1] * t_baseline_bin.shape[0])

        # flatten the data for SVM
        X = t_all_bin.reshape(t_all_bin.shape[0], -1)

        accuracy_scores = []
        surrogate_scores = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train SVM
            clf = svm.LinearSVC(max_iter=n_iter, tol=tol)
            clf.fit(X_train, y_train)

            # calculate accuracy
            accuracy = f1_score(y_test, clf.predict(X_test))
            accuracy_scores.append(accuracy)

            # Surrogate analysis
            for _ in tqdm(range(n_shuffles)):
                y_train_shuffle = shuffle(y_train)
                clf.fit(X_train, y_train_shuffle)
                surrogate_accuracy = f1_score(y_test, clf.predict(X_test))
                surrogate_scores.append(surrogate_accuracy)

        # calculate average and standard deviation of accuracy for the current time point
        avg_performance.append(np.mean(accuracy_scores))
        std_performance.append(np.std(accuracy_scores))
        surrogate_performance.append(surrogate_scores)

    avg_performance = np.asarray(avg_performance)
    std_performance = np.asarray(std_performance)
    surrogate_performance = np.asarray(surrogate_performance)

    return avg_performance, std_performance, surrogate_performance

def train_SVM_baseline(t_sound_day, t_baseline_day, time_bin_size, n_splits=5, n_shuffles=100, n_iter=1000, tol=1e-4):
    # Create time points based on time bin size
    time_points = range(0, t_sound_day.shape[2], time_bin_size)

    # initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits)

    avg_performance = []
    std_performance = []
    surrogate_performance = []
    svm_coefficients = []  # to store the SVM coefficients

    # for each time bin, train a separate SVM classifier
    for i in time_points:
        print(f'Computing time bin {int(i/time_bin_size)}/{len(time_points)}')

        # get the data for this time bin
        t_sound_bin = t_sound_day[:, :, i:i+time_bin_size].mean(axis=2)  # taking average over time_bin dimension
        t_baseline_bin = t_baseline_day.mean(axis=2)  # taking average over time_bin dimension

        # concatenate the sound and baseline ndarrays for this time bin (trials, components)
        t_all_bin = np.concatenate((t_sound_bin, t_baseline_bin), axis=0)

        # prepare the label vector
        y = np.array([0] * t_sound_bin.shape[0] + [1] * t_baseline_bin.shape[0])

        # flatten the data for SVM (trials, components)
        X = t_all_bin.reshape(t_all_bin.shape[0], -1)

        accuracy_scores = []
        surrogate_scores = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train SVM
            clf = svm.LinearSVC(max_iter=n_iter, tol=tol)
            clf.fit(X_train, y_train)

            # calculate accuracy
            accuracy = f1_score(y_test, clf.predict(X_test))
            accuracy_scores.append(accuracy)

            svm_coefficients.append(clf.coef_)  # store the SVM coefficients
            
            # Surrogate analysis
            for _ in tqdm(range(n_shuffles)):
                y_train_shuffle = shuffle(y_train)
                clf.fit(X_train, y_train_shuffle)
                surrogate_accuracy = f1_score(y_test, clf.predict(X_test))
                surrogate_scores.append(surrogate_accuracy)
            

        # calculate average and standard deviation of accuracy for the current time point
        avg_performance.append(np.mean(accuracy_scores))
        std_performance.append(np.std(accuracy_scores))
        surrogate_performance.append(surrogate_scores)
        
    svm_coefficients = np.array(svm_coefficients)
    avg_performance = np.asarray(avg_performance)
    std_performance = np.asarray(std_performance)
    surrogate_performance = np.asarray(surrogate_performance)

    return avg_performance, std_performance, surrogate_performance, svm_coefficients

def moving_average(a, n=3):
    if n > 1:
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    else:
        return a
    
def plot_decoding_performance(avg_performance, std_performance, color, fps=50, time_bin_size=5, window_size=5, label=None, ypos=1):
    # Apply moving average
    avg_performance_smooth = moving_average(avg_performance, window_size)
    std_performance_smooth = moving_average(std_performance, window_size)

    # Compute bin_centers_sec_smooth after smoothing
    num_bins = len(avg_performance_smooth)
    bin_centers_sec = np.arange(avg_performance.shape[0])*time_bin_size/fps - 100/50  # First calculate for the original performance data
    
    if window_size > 1:
        bin_centers_sec_smooth = bin_centers_sec[:num_bins]  # Then adjust for the smoothing
    else:
        bin_centers_sec_smooth = bin_centers_sec  # No smoothing, so they are the same

    # Ensure the lengths match after smoothing
    bin_centers_sec_smooth = bin_centers_sec_smooth[:len(avg_performance_smooth)]

    plt.plot(bin_centers_sec_smooth, avg_performance_smooth, color = color, zorder = 1)
    plt.fill_between(bin_centers_sec_smooth, avg_performance_smooth - std_performance_smooth,
                     avg_performance_smooth + std_performance_smooth, color = color, alpha=0.2, zorder = 1)
    
    plt.axhline(y=0.5, color="black", linestyle=(0, (1, 5))) #, label='chance level')
    # Add the axvspan for CS+ sound presentation
    plt.axvspan(0, 2, color='lightgray', alpha=1, zorder=0)  # label="Sound")
    # Add the axvspan for Reward period
    plt.axvspan(2 + 1, 2 + 2, color='#B5DBA8', alpha=1, zorder=0) # label="Reward")

    plt.xlabel('Time from sound onset (s)')
    plt.ylabel('Accuracy (%)')

    # Modify y-tick labels to display as multiples of 100 without the % symbol
    formatter = mticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*100))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Set y-limit
    # plt.ylim(0, 1)
    
    # Get the current axes
    ax = plt.gca()
    # Modify the spines of the current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a colored label at the top left corner
    plt.text(0.02, ypos, label, transform=plt.gca().transAxes, color=color, fontsize=12, va='top')
