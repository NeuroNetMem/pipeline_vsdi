import numpy as np
from tqdm import tqdm
import tifffile


def read_raw_file(filename, h, w, start_frame=0,end_frame=None, precision='>f'): 

    """
    Read a slice of frames from a raw binary file and reshape into a 3D array.

    This function reads a specified range of frames from a raw binary file, interprets
    the data according to the specified precision, and reshapes it into a 3D numpy array
    representing the frames.

    Parameters:
    filename (str): The name of the raw binary file to read.
    h (int): The height of each frame.
    w (int): The width of each frame.
    start_frame (int, optional): The starting frame index (default is 0).
    end_frame (int, optional): The ending frame index (default is None, which indicates
                              the last frame of the slice).
    precision (str, optional): The data precision format used in the binary file
                              (default is '>f', which represents big-endian 4-byte float).

    Returns:
    numpy.ndarray: A 3D numpy array containing the frames, with dimensions (h, w, n_frames).

    Note:
    - The 'order="F"' argument is used to reshape the data in Fortran order, which is column-major
      like the memory layout of numpy arrays.

    Example:
    frame_slice = read_raw_file('file.raw', h=256, w=256, start_frame=0, end_frame=100,
                                      precision='>f')
    """

    n_frames = end_frame - start_frame
    element_size = np.dtype(precision).itemsize # byte size of a single pixel
    offset_bytes = h * w * start_frame *element_size # offset for the file stream
    
    elements_to_read = h*w*n_frames # number of elements to read from stream
    
    with open(filename, "rb") as filestream:
        filestream.seek(offset_bytes) # starts from computed offset
        img = np.fromfile(filestream, dtype=precision,count=elements_to_read)
    img = np.reshape(img, [h, w, n_frames], order="F")
    
    return img

def read_vsdi_file(filename,start_frame = 0, end_frame=None, precision='>f'):
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
        start_frame (int,optional): Frame to start reading the video from. Default value is 0.
        end_frame (int,optional): Frame to end the video reading at. If not provided, the video will be read
            until the end.

    Returns:
        video (ndarray): A NumPy array containing the video read from the VSDI file with shape (h x w x n_frames).

    Raises:
        FileNotFoundError: If the specified filename does not exist.

    Example:
        filename = 'block1_A03_day2_75026fr_125Hz_480x300_Bandpass_Cheby_high_0.5_9FHz_D_F0_.raw'
        video = read_vsdi_file(filename)
    """

    if end_frame is None:
        end_frame = int(filename.split('x')[0].split('fr')[0].split('_')[-1])

        

    w = int(filename.split('x')[1].split('_')[0])
    h = int(filename.split('x')[0].split('_')[-1])
    video = read_raw_file(filename, h, w,
                                start_frame = start_frame,
                                end_frame = end_frame,
                                precision=precision)

    return video


def read_and_downsample_vsdi(filename, stride, start_frame = 0, end_frame= None,
                             precision='>f',batch_size=1000):
    """
    Read from a VSDI .raw file, and downsample images with specified stride.

    This function reads a specified range of frames from a VSDI file,
    downsampling the frames using a given stride, and returns the downsampled frames
    in a single array.

    Parameters:
    filename (str): The name of the VSDI file to read.
    stride (int): The downsampling stride for both width and height of frames.
    start_frame (int, optional): The starting frame index (default is 0).
    end_frame (int, optional): The ending frame index (default is None, which indicates
                              the last frame of the file).
    precision (str, optional): The data precision format to read from the file
                              (default is '>f', which represents big-endian 4-byte float).
    batch_size (int, optional): The number of frames to read and process in each batch
                              (default is 1000). This roughly correspond to a 1Gb memory load at
                              all times. If more memory is available, batch_size can be increased.

    Returns:
    numpy.ndarray: A 3D numpy array containing the downsampled frames stacked along the
                  third dimension.

    Note:
    - The downsampling is performed by skipping pixels in both width and height of each frame.
    - The resulting frames are stored in a single array, with the third dimension containing
      the downsampled frames.
    - The function uses the provided batch size to read and process frames in
      smaller chunks, to be able to work with larger-than-memory files.

    Example:
    downsampled_video = read_and_downsample_vsdi('Block1_A02_d7_mc_27718fr_125Hz_480x300_Bandpass_Cheby_high_0.5_9FHz_D_F0_.raw', 
                                                  stride=2, start_frame=0,end_frame=100, precision='>f', batch_size=500)
    """

    

    out_video = []
    
    element_size = np.dtype(precision).itemsize

    w = int(filename.split('x')[1].split('_')[0])
    h = int(filename.split('x')[0].split('_')[-1])
    
    if end_frame is None:
        end_frame = int(filename.split('x')[0].split('fr')[0].split('_')[-1])


    n_frames = end_frame-start_frame

    start_frame = 0
    n_batches = int(np.ceil(n_frames/batch_size))

    for i in tqdm(range(n_batches)):
        end_frame = start_frame + batch_size
        if end_frame > n_frames:
            end_frame = n_frames
            
        video = read_raw_file(filename, h, w,
                                start_frame = start_frame,
                                end_frame = end_frame,
                                precision=precision)
        out_video.append(video[::stride,::stride,:])
        start_frame = end_frame

    out_video = np.dstack(out_video)


    return out_video


def read_mask_tifffile(tif_file):
    """
    Read a TIF image file containing a mask and return its data as a NumPy array, 
    transposed to match the vsdi video shape (h,w).

    Parameters:
    tif_file (str): The path to the TIF image file.

    Returns:
    numpy.ndarray: A NumPy array representing the mask image data, transposed to match vsdi video(h,w).
    """


    
    return tifffile.imread(tif_file).T


def decode_b64_file(filepath, verbose=False):
    """
    Decode a file containing Base64-encoded binary data packets, and extract analog and digital channel information.

    Parameters:
        filepath (str): The path to the Base64-encoded file containing binary data packets.
        verbose (bool, optional): If True, enable verbose mode to display progress and additional information.
                                  Default is False.

    Returns:
        dict: A dictionary containing the extracted data organized into different channel groups.
              The dictionary has the following keys:
                  - 'analog': A 2D array containing analog channel data for each packet (packet x channel).
                  - 'digitalIn': A 2D array containing digital input channel data for each packet (packet x channel).
                  - 'digitalOut': A 2D array containing digital output channel data for each packet (packet x channel).
                  - 'startTS': An array containing the start timestamp for each packet.
                  - 'transmitTS': An array containing the transmit timestamp for each packet.
                  - 'longVar': A 2D array containing state variables data for each packet (packet x channel).
                  - 'packetNums': An array containing packet IDs for each packet.

    Notes:
        - The function expects the data packets to be Base64-encoded and packed with the COBS protocol.
        - The function uses the hard-coded 'DataPacketDesc' to parse the binary data.
    """

    bp = filepath

    # Format package
    DataPacketDesc = {'type': 'B',
                      'size': 'B',
                      'crc16': 'H',
                      'packetID': 'I',
                      'us_start': 'I',
                      'us_end': 'I',
                      'analog': '8H',
                      'states': '8l',
                      'digitalIn': '2H',
                      'digitalOut': '3B',
                      'padding': 'x'}

    DataPacket = namedtuple('DataPacket', DataPacketDesc.keys())
    DataPacketStruct = '<' + ''.join(DataPacketDesc.values())
    DataPacketSize = struct.calcsize(DataPacketStruct)

    # package with non-digital data
    dtype_no_digital = [
        ('type', np.uint8),
        ('size', np.uint8),
        ('crc16', np.uint16),
        ('packetID', np.uint32),
        ('us_start', np.uint32),
        ('us_end', np.uint32),
        ('analog', np.uint16, (8, )),
        ('states', np.uint32, (8, ))]

    # DigitalIn and DigitalOut
    dtype_w_digital = dtype_no_digital + \
        [('digital_in', np.uint16, (16, )), ('digital_out', np.uint16, (16, ))]

    # Creating arrat with all the data (differenciation digital/non digital)
    np_DataPacketType_noDigital = np.dtype(dtype_no_digital)
    np_DataPacketType_withDigital = np.dtype(dtype_w_digital)
    # Unpack the data as done on the teensy commander code

    # function to count the packet number
    def count_lines(fp):
        def _make_gen(reader):
            b = reader(2**17)
            while b:
                yield b
                b = reader(2**17)
        with open(fp, 'rb') as f:
            count = sum(buf.count(b'\n') for buf in _make_gen(f.raw.read))
        return count

    num_lines = count_lines(bp)
    log_duration = num_lines/1000/60
    if verbose:
        print(bp)
        print(f'{num_lines} packets, ~{log_duration:0.2f} minutes')

    def unpack_data_packet(dp):
        s = struct.unpack(DataPacketStruct, dp)
        up = DataPacket(type=s[0], size=s[1], crc16=s[2], packetID=s[3], us_start=s[4], us_end=s[5],
                        analog=s[6:14], states=s[14:22], digitalIn=s[22], digitalOut=s[23], padding=None)

        return up

    # Decode and create new dataset
    data = np.zeros(num_lines, dtype=np_DataPacketType_withDigital)
    print(len(data))
    non_digital_names = list(np_DataPacketType_noDigital.names)

    with open(bp, 'rb') as bf:
        for nline, line in enumerate(tqdm(bf, total=num_lines, disable=not verbose)):
            bl = cobs.decode(base64.b64decode(line[:-1])[:-1])
            dp = unpack_data_packet(bl)
            data[non_digital_names][nline] = np.frombuffer(
                bl[:-8], dtype=np_DataPacketType_noDigital)
            digital_arr = np.frombuffer(bl[-8:], dtype=np.uint8)
            data[nline]['digital_in'] = np.hstack(
                [np.unpackbits(digital_arr[1]), np.unpackbits(digital_arr[0])])
            data[nline]['digital_out'] = np.hstack(
                [np.unpackbits(digital_arr[3]), np.unpackbits(digital_arr[2])])

    # Check for packetID jumps
    jumps = np.unique(np.diff(data['packetID']))

    # assert(len(jumps) and jumps[0] == 1)
    data['digital_in'] = np.flip(data['digital_in'], 1)
    data['digital_out'] = np.flip(data['digital_out'], 1)
    decoded = {"analog": data['analog'], "digitalIn": data['digital_in'], "digitalOut": data['digital_out'],
               "startTS": data['us_start'], "transmitTS": data['us_end'], "longVar": data['states'], "packetNums": data['packetID']}

    return decoded


def make_channel_dict(channel_data, name_dict=None):
    """
    Creates a dictionary containing channel data organized according to the provided 'name_dict'.

    Parameters:
        channel_data (dict): A dictionary containing the channel data. It should have the following keys:
                             'analog', 'digitalIn', 'digitalOut', 'longVar', 'startTS', 'transmitTS', 'packetNums'.
        name_dict (dict, optional): A dictionary that maps channel groups to their respective channel names.
                                    If not provided, default names will be used.
                                    Default name_dict:
                                    {
                                        'analog': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'],
                                        'digitalIn': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8',
                                                      'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16'],
                                        'digitalOut': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8',
                                                       'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16'],
                                        'longVar': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
                                    }

    Returns:
        dict: A two-level dictionary containing the channel data organized as per the 'name_dict'.
              The first level keys represent channel groups, and the second level keys represent channel names.
              The values contain the corresponding channel data for each group and name.
              Additionally, 'startTS', 'transmitTS', and 'packetNums' data are also included in the output.

    """

    output_dict = {}

    default_name_dict = {
        'analog': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'],
        'digitalIn': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16'],
        'digitalOut': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16'],
        'longVar': ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
    }

    # if no naming is provided, uses default names
    if name_dict == None:
        name_dict = default_name_dict

    # builds a two level dictionary with channel group and channel name from name dictionary
    for key in name_dict.keys():
        ch_names = name_dict[key]
        ch_group = channel_data[key]
        output_dict[key] = {}
        for i, n in enumerate(ch_names):
            output_dict[key][n] = ch_group[:, i]

    # append the rest of metadata
    output_dict['startTS'] = channel_data['startTS']
    output_dict['transmitTS'] = channel_data['transmitTS']
    output_dict['packetNums'] = channel_data['packetNums']

    return output_dict

# to implement


def read_vr_log():
    pass


def read_lfp_file():

    pass
