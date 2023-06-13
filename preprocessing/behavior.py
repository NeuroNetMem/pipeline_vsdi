# Data management
import numpy as np

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

def compute_and_add_lick_rate(df, fps):
    """
    Lick Rate (Hz) per frame using X dataframe version
    """
    # Group by 'Animal', 'Day', 'Session' and apply rolling window calculation
    df['Lick Rate (Hz)'] = df.groupby(['Animal', 'Day', 'Session'])['Lick'].transform(lambda x: x.rolling(fps, min_periods=1).sum())
    return df

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
