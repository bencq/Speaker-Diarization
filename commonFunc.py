import argparse
import librosa
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dealWav(wavData):
    intervals = librosa.effects.split(wavData, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wavData[sliced[0]:sliced[1]])
    return np.array(wav_output)