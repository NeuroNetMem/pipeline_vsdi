import numpy as np


def read_data_fields(atc_data):
    data = atc_data['Data_OE'][0][0]
    names = np.asarray(atc_data['Data_OE'][0][0].dtype.names)
    for i, k in enumerate(names):
        print(f'{k}:{data[i].shape}')


def extract_behavioural_data(atc_data):
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
    names = np.asarray(atc_data['Data_OE'][0][0].dtype.names)
    lfp_name = 'HPC_LFP'
    if not lfp_name in names:
        raise TypeError(f'{lfp_name} not found in data structure')

    lfp_idx = np.where(names == lfp_name)[0][0]
    lfp_data = atc_data['Data_OE'][0][0][lfp_idx][0]
    lfp = np.vstack([i.flatten() for i in lfp_data])

    return lfp
