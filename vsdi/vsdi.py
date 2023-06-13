from pathlib import Path
from scipy.io import loadmat
import h5py

# from loaders import *
# from pipeline_vsdi.preprocessing.utils import *

class DataLoader:
    """
    Handling VSDI data full dataset using HDF5 files

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file to be loaded.
    datapath : str
        Path to the directory containing the VSI data files.
    """
    def __init__(self, filepath, datapath):
        """Initialize the DataLoader."""
        self.filepath = filepath
        self.datapath = Path(datapath)

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
        """
        Exit method for the DataLoader context manager.

        Parameters
        ----------
        type : type
            The type of the exception, if an exception occurred.
        value : exception or None
            The exception that occurred, if any.
        traceback : traceback or None
            The traceback object associated with the exception, if any.
        """
        self.file.close()

    def load_data(self, animals, days, sessions):
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
        total_animals = len(animals)
        total_days = len(days)
        total_sessions = len(sessions)
        
        for animal in animals:
            animal_group = self.file.require_group(animal)

            for day in days:
                day_group = animal_group.require_group(day)

                # Load mask file for each day
                mask = loadmat(self.datapath.joinpath(f'{animal}/{day}/vsdi_mask.mat'))['mask']
                if 'mask' in day_group:
                    del day_group['mask']
                day_group.create_dataset('mask', data=mask)

                for session in sessions:
                    session_group = day_group.require_group(session)

                    # Load matlab behavioural file
                    atc = loadmat(self.datapath.joinpath(f'{animal}/{day}/{session}.mat'))

                    # Extract dictionary
                    b_data = loaders.extract_behavioural_data(atc)
                    # Create ndarray design matrix
                    X_matrix = make_design_matrix(b_data, event_sequences)

                    if 'behavioral' in session_group:
                        del session_group['behavioral']
                    session_group.create_dataset('behavioral', data=X_matrix)

                    # Load VSDI
                    vsdi = loadmat(self.datapath.joinpath(f'{animal}/{day}/vsdi_{session}.mat'))['vsdi_data']

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
                    group.parent.create_dataset('vsdi_clean', data=vsdi_data_cleaned)
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
