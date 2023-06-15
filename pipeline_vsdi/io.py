import numpy as np


# raw file readers

def imreadallraw(filename, x, y, nFrame, precision):
    fid0 = open(filename, "rb")
    img0 = np.fromfile(fid0, dtype=precision)
    fid0.close()
    img0 = np.reshape(img0, [x, y, nFrame], order="F")
    return img0


def imreadallraw2(filename, precision):
    features = filename.split('_')
    n_frames = int(float(features[6][:-2]))
    w, h = features[8].split('x')
    w = int(w)
    h = int(h)

# log file readers
