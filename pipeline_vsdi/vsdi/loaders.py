"""
This module contains functions to read and extract data from ATC (Assistive Technology Comparison) data objects.

The ATC data objects are expected to be a dictionary like objects that represent the contents of .mat files, 
which contain various fields of data related to behavioural experiments. 

The key functions in the module include:

- read_data_fields: This function reads and prints the names and shapes of data fields in the given ATC data object.

- extract_behavioural_data: This function extracts specific behavioural data fields from an ATC data object.

- extract_lfp_data: This function extracts the local field potential (LFP) data from the given ATC data object.

Please refer to the individual function docstrings for more detailed information on their usage and output.

Example usage:

>>> import scipy.io
>>> atc_data = scipy.io.loadmat('data.mat')  # Load .mat file
>>> read_data_fields(atc_data)
>>> extracted_data = extract_behavioural_data(atc_data)
>>> lfp_data = extract_lfp_data(atc_data)
"""

import numpy as np


def read_data_fields(atc_data):
    """
    Reads and prints the names and shapes of data fields in the given ATC data object.

    This function reads the fields from the input ATC data object and then prints out each field's name and shape.

    Parameters
    ----------
    atc_data : dictionary
        The input ATC data object which contains multiple fields. It's expected to be a dictionary 
        like object (such as .mat file) that contains data fields.

    Returns
    -------
    None

    Prints
    ------
    For each field in the ATC data object, its name and shape are printed to the console.

    Examples
    --------
    >>> atc_data = loadmat('data.mat')  # Load .mat file with scipy.io.loadmat
    >>> read_data_fields(atc_data)
    """
    data = atc_data['Data_OE'][0][0]
    names = np.asarray(atc_data['Data_OE'][0][0].dtype.names)
    for i, k in enumerate(names):
        print(f'{k}:{data[i].shape}')


def extract_behavioural_data(atc_data):
    """
    Extracts specific behavioural data fields from an ATC data object.

    This function reads fields from the input ATC data, checks if the required fields are present, 
    and then extracts and flattens the required fields.

    Parameters
    ----------
    atc_data : dictionary
        The input ATC data object which contains multiple fields. It's expected to be a dictionary 
        like object (such as .mat file) that contains data fields.

    Returns
    -------
    extracted_data : dictionary
        A dictionary containing the extracted data for each of the required fields. Each field 
        is flattened and represented as a 1-D array.

    Raises
    ------
    TypeError
        If a required field is not found in the ATC data object, a TypeError is raised indicating 
        the missing field.

    Notes
    -----
    The fields to be extracted are defined in the `to_extract` variable within the function. 
    Currently, these include 'CSp', 'CSn', 'Lick', and 'frames'. If other fields are needed, 
    this list should be updated accordingly.

    Examples
    --------
    >>> atc_data = loadmat('data.mat')  # Load .mat file with scipy.io.loadmat
    >>> extracted_data = extract_behavioural_data(atc_data)
    """
    # reads daa fields in input .mat file
    names = np.asarray(atc_data['Data_OE'][0][0].dtype.names)
    # fields to extract
    to_extract = ['CSp', 'CSn', 'Lick', 'frames']
    extracted_data = {}
    for k in to_extract:
        # check if field is in data
        if not (k in names):
            raise TypeError(f' field name {k} not found in data structure')
        # find index corresponding to field name
        idx = np.where(names == k)[0][0]
        # extract data, flattens them
        extracted_data[k] = atc_data['Data_OE'][0][0][idx].flatten()

    return extracted_data


def extract_lfp_data(atc_data):
    """
    Extracts the local field potential (LFP) data from the given ATC data object.

    This function reads the input ATC data object, checks if it contains the 'HPC_LFP' field, 
    and then extracts and returns the 'HPC_LFP' data. The 'HPC_LFP' data are flattened and stacked vertically.

    Parameters
    ----------
    atc_data : dictionary
        The input ATC data object which contains multiple fields including 'HPC_LFP'. It's expected to be a 
        dictionary like object (such as .mat file) that contains data fields.

    Returns
    -------
    lfp : ndarray
        A 2-D numpy array that contains the extracted 'HPC_LFP' data. Each sub-array is flattened and 
        the sub-arrays are stacked vertically.

    Raises
    ------
    TypeError
        If the 'HPC_LFP' field is not found in the ATC data object.

    Examples
    --------
    >>> atc_data = loadmat('data.mat')  # Load .mat file with scipy.io.loadmat
    >>> lfp_data = extract_lfp_data(atc_data)
    """
    names = np.asarray(atc_data['Data_OE'][0][0].dtype.names)
    lfp_name = 'HPC_LFP'
    if not lfp_name in names:
        raise TypeError(f'{lfp_name} not found in data structure')

    lfp_idx = np.where(names == lfp_name)[0][0]
    lfp_data = atc_data['Data_OE'][0][0][lfp_idx][0]
    lfp = np.vstack([i.flatten() for i in lfp_data])

    return lfp
