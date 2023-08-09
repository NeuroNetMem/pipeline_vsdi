"""
This module provides a `VSDISession` class for handling VSDI (Voltage-Sensitive Dye Imaging) data, behaviour data
and metadata for a given experimental session in a single interface.

"""
import os
from dataclasses import dataclass
import h5py
import numpy as np

import io


@dataclass
class VsdiSession:

    metadata: dict = None
    vsdi_video: np.array = None
    vsdi_time: np.array = None
    vsdi_mask: np.array = None
    vr_data: dict = None
    lfp: dict = None

    def initialize_from_raw_data(self, folder):

        # read vsdi and b64 files from folder

        try:
            self.vsdi_video = io.read_vsdi_file(vsdi_file)
        except FileNotFoundError:
            print('Vsdi video not found')

        try:
            self.vr_data = io.read_vr_log(vr_file)
        except FileNotFoundError:
            print('VR logfile not found')

        try:
            self.lfp = io.read_lfp_file()
        except FileNotFoundError:
            print('LFP file not found')

    def initialize_from_hdf5_file(self, file):
        pass

    def load_vsdi_fromfile(self, video_file):
        self.vsdi_video = io.read_vsdi_file(video_file)
        return

    def load_mask_fromfile(self, mask_file):
        self.vsdi_mask = np.load(mask_file)
        return

    def load_vr_fromfile(self, logfile):
        self.vr_data = io.read_vr_log()

    def load_lfp_fromfile(self, lfp_file):
        self.lfp = io.read_lfp_file()

    def add_metadata(self, metadata_dict):
        required_keys = ['animal', 'session', 'vsdi_fps']

        if all(key in metadata_dict for key in required_keys):
            self.metadata = metadata_dict
        else:
            missing_keys = [
                key for key in required_keys if key not in metadata_dict]
            raise ValueError(f"Missing required keys: {missing_keys}")

        return

    def get_flat_vsdi(self):
        '''
        Returns the vsdi video if a flattened format, ready for Machine Learning applications.
        The video is subsetted with the cortical mask, flattened and transpose to have shape (n_samples x n_features)
        '''
        flat_vsdi = self.vsdi_video[self.vsdi_mask, :].T

        return flat_vsdi

    def set_vsdi_from_flat(self, flat_vsdi):
        pass

    def to_hdf5():
        pass

def save_vsdi_session(filename, vsdi_session):
    with h5py.File(filename, 'w') as file:
        # Create a group for VsdiSession
        session_group = file.create_group('VsdiSession')
        
        # Save metadata
        metadata_group = session_group.create_group('metadata')
        for key, value in vsdi_session.metadata.items():
            metadata_group[key] = value
        
        # Save arrays using datasets
        session_group.create_dataset('vsdi_video', data=vsdi_session.vsdi_video)
        session_group.create_dataset('vsdi_time', data=vsdi_session.vsdi_time)
        session_group.create_dataset('vsdi_mask', data=vsdi_session.vsdi_mask)
        
        # Save VR data
        vr_data_group = session_group.create_group('vr_data')
        for key, value in vsdi_session.vr_data.items():
            vr_data_group[key] = value
        
        # Save LFP data
        lfp_group = session_group.create_group('lfp')
        for key, value in vsdi_session.lfp.items():
            lfp_group[key] = value

def load_vsdi_session(filename):
    with h5py.File(filename, 'r') as file:
        session_group = file['VsdiSession']
        
        metadata = dict(session_group['metadata'])
        vsdi_video = session_group['vsdi_video'][:]
        vsdi_time = session_group['vsdi_time'][:]
        vsdi_mask = session_group['vsdi_mask'][:]
        
        vr_data = dict(session_group['vr_data'])
        lfp = dict(session_group['lfp'])
        
        vsdi_session = VsdiSession(metadata, vsdi_video, vsdi_time, vsdi_mask, vr_data, lfp)
        return vsdi_session



class DataLoader:
    """
    Handling VSDI data full dataset using HDF5 files

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file to be loaded.
    """

    def __init__(self, filepath):
        """Initialize the DataLoader."""
        self.filepath = filepath

    def __enter__(self):
        """
        Enter method for the DataLoader context manager.

        Returns
        -------
        self : DataLoader
            The DataLoader instance.
        """
        self.file = h5py.File(self.filepath, 'a')
        return self

    def __exit__(self, type, value, traceback):
        """Exit method for the DataLoader context manager."""
        self.file.close()

    def load_data(self, datapath, animals, days, sessions, event_sequences):
        """
        Load VSI data into the HDF5 file.

        Parameters
        ----------
        animals : list
            List of animal names.
        days : list
            List of day names.
        sessions : list
            List of session names.
        """
        for animal in animals:
            animal_group = self.file.require_group(animal)

            for day in days:
                day_group = animal_group.require_group(day)

                # Load mask file for each day
                mask = loadmat(datapath.joinpath(
                    f'{animal}/{day}/vsdi_mask.mat'))['mask']
                if 'mask' in day_group:
                    del day_group['mask']
                day_group.create_dataset('mask', data=mask)

                for session in sessions:
                    session_group = day_group.require_group(session)

                    # Load matlab behavioural file
                    atc = loadmat(datapath.joinpath(
                        f'{animal}/{day}/{session}.mat'))

                    # Extract dictionary
                    b_data = loaders.extract_behavioural_data(atc)
                    # Create ndarray design matrix
                    X_matrix = make_design_matrix(b_data, event_sequences)

                    if 'behavioral' in session_group:
                        del session_group['behavioral']
                    session_group.create_dataset('behavioral', data=X_matrix)

                    # Load VSDI
                    vsdi = loadmat(datapath.joinpath(
                        f'{animal}/{day}/vsdi_{session}.mat'))['vsdi_data']

                    if 'vsdi' in session_group:
                        del session_group['vsdi']
                    session_group.create_dataset('vsdi', data=vsdi)

                    # Extract LFP data
                    lfp = loaders.extract_lfp_data(atc)

                    if 'lfp' in session_group:
                        del session_group['lfp']
                    session_group.create_dataset('lfp', data=lfp)

    def clean_vsdi(self, animal=None, day=None, session=None, nsigma=3):
        """
        Clean VSDI data by removing outliers.

        Parameters
        ----------
        animal : str, optional
            Animal name.
        day : str, optional
            Day name.
        session : str, optional
            Session name.
        nsigma : int, optional
            Number of standard deviations for outlier removal.
        """
        def clean(vsdi, mask):
            # Normalize vsdi
            vsdi_norm = normalize_vsdi(vsdi, mask)
            # Clean outliers
            vsdi_clean = clean_outliers(vsdi_norm, nsigma)
            # Clean outliers by mean interpolation
            return vsdi_clean

        def apply_clean(group, mask):
            if isinstance(group, h5py.Dataset):
                # Assuming 'vsdi', 'animal', 'day', 'session' are keys, not attributes
                if group.name.split('/')[-1] == 'vsdi':
                    vsdi_data = group[...]
                    vsdi_data_cleaned = clean(vsdi_data, mask)
                    # Create a new dataset for cleaned VSDI data
                    if 'vsdi_clean' in group.parent:
                        del group.parent['vsdi_clean']
                    group.parent.create_dataset(
                        'vsdi_clean', data=vsdi_data_cleaned)
            else:
                for key in group:
                    apply_clean(group[key], mask)

        if animal is not None and day is not None and session is not None:
            mask = self.file[f"{animal}/{day}"]['mask'][...]
            group = self.file[f"{animal}/{day}/{session}"]
            apply_clean(group, mask)
        elif animal is not None and day is not None:
            mask = self.file[f"{animal}/{day}"]['mask'][...]
            group = self.file[f"{animal}/{day}"]
            apply_clean(group, mask)
        elif animal is not None:
            for day in self.file[animal].keys():
                mask = self.file[f"{animal}/{day}"]['mask'][...]
                group = self.file[animal][day]
                apply_clean(group, mask)
        else:
            for animal in self.file.keys():
                for day in self.file[animal].keys():
                    mask = self.file[f"{animal}/{day}"]['mask'][...]
                    group = self.file[animal][day]
                    apply_clean(group, mask)

    def get_data(self, animal=None, day=None, session=None, type=None):
        """
        Get data from the HDF5 file.

        Parameters
        ----------
        animal : str, optional
            Animal name.
        day : str or list, optional
            Day name or list of day names.
        session : str, optional
            Session name.
        type : str, optional
            Type of data.

        Returns
        -------
        result : list
            List of tuples containing the group name and data array.
        """
        days = [day] if isinstance(day, str) else day

        def search(group):
            result = []
            for key in group:
                if isinstance(group[key], h5py.Group):
                    result.extend(search(group[key]))
                elif ((animal is None or group.name.split('/')[1] == animal) and
                      (days is None or group.name.split('/')[2] in days) and
                      (session is None or group.name.split('/')[3] == session) and
                      (type is None or key == type)):
                    result.append((group.name.split('/')[1:], group[key][...]))
            return result

        return search(self.file)
