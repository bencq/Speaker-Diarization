import os
import sys
import random
import librosa
import argparse
import numpy as np
import soundfile

parser = argparse.ArgumentParser()

parser.add_argument('--speakerPath0', required=True, type=str)  # under which directory contains wavs
parser.add_argument('--num0', default=5, type=int)
parser.add_argument('--speakerPath1', required=True, type=str)
parser.add_argument('--num1', default=5, type=int)
parser.add_argument('--wavOutPath', required=True, type=str)

args = parser.parse_args()

_sr = 16000

if __name__ == '__main__':

    assert args.wavOutPath.endswith('.wav')
    txtOutPath = args.wavOutPath[:-len('.wav')] + ".txt"

    wavNames0 = [name for name in os.listdir(args.speakerPath0) if name.endswith('.wav')]
    wavNames1 = [name for name in os.listdir(args.speakerPath1) if name.endswith('.wav')]

    intervals = []  # list of [startFrameInd, endFrameInd, clusterId]
    totalData = []  # list of [label(0 or 1), data]

    for ind in range(args.num0):
        wavPath = os.path.join(args.speakerPath0, wavNames0[ind])
        data, sr = librosa.load(wavPath, sr=None)
        assert sr == _sr
        assert len(data.shape) == 1
        totalData.append( [0, data] )

    for ind in range(args.num1):
        wavPath = os.path.join(args.speakerPath1, wavNames1[ind])
        data, sr = librosa.load(wavPath, sr=None)
        assert sr == _sr
        assert len(data.shape) == 1
        totalData.append( [1, data] )

    random.shuffle(totalData)

    curFrame = 0
    for pair in totalData:
        intervals.append([curFrame, curFrame + pair[1].shape[0] - 1, pair[0]])
        curFrame += pair[1].shape[0]

    wavOut = np.concatenate([pair[1] for pair in totalData])
    print(wavOut.shape)
    print(curFrame)

    # librosa.output.write_wav(args.wavOutPath, wavOut, _sr)
    soundfile.write(args.wavOutPath, data=wavOut, samplerate=_sr, format="WAV")
    txtFile = open(txtOutPath, mode='w')
    for inter in intervals:
        txtFile.write(str(inter[0]) + " " + str(inter[1]) + " " + str(inter[2]) + "\n")
    txtFile.close()




