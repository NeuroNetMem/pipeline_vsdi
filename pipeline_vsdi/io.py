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
