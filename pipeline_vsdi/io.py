import numpy as np


def read_vsdi_file(filename, precision='>f'):
    """
    Reads a VSDI (Voltage-Sensitive Dye Imaging) file and returns the video as a numpy array.
    The name of the file has to be in the following format:
    'session_info_NFfr_FRHz_WWxHH_preprocessing_info.raw'
    where:
    NF: number of frames
    FR: frame rate
    WW: frame width in pixels
    HH: frame height in pixels



    Parameters:
        filename (str): The name of the VSDI file to be read.
        precision (str, optional): The precision of the data stored in the file.
            It determines how the data is interpreted. The default value is '>f'
            which stands for big-endian single-precision floating-point format.

    Returns:
        video (ndarray): A NumPy array containing the video read from the VSDI file.

    Raises:
        FileNotFoundError: If the specified filename does not exist.

    Example:
        filename = 'block1_A03_day2_75026fr_125Hz_480x300_Bandpass_Cheby_high_0.5_9FHz_D_F0_.raw'
        video = read_vsdi_file(filename)
    """

    n_frames = int(filename.split('x')[0].split('fr')[0].split('_')[-1])

    w = int(filename.split('x')[0].split('_')[-1])
    h = int(filename.split('x')[1].split('_')[0])
    video = read_raw_file(filename, h, w, n_frames, precision=precision)

    return video


def read_raw_file(filename, h, w, n_frames, precision='>f'):
    with open(filename, "rb") as filestream:
        img = np.fromfile(filestream, dtype=precision)
    img = np.reshape(img, [h, w, n_frames], order="F")
    return img

# to implement


def read_vr_log():
    pass


def read_lfp_file():
    pass
