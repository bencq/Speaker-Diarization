"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import sys

import librosa
import numpy as np

import uisrnn

sys.path.append('ghostvlad')
sys.path.append('visualization')
import toolkits
import model as spkModel
import os
import commonFunc
import soundfile

from viewer import PlotDiar

# ===========================================
#        Parse the argument
# ===========================================
import argparse

parser = argparse.ArgumentParser()
# # set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

# set record wav path
parser.add_argument('--wavPath', required=True, type=str)
# set uis-rnn path
parser.add_argument('--modelPath', default='pretrained/saved_model.uisrnn_benchmark', type=str)
# decide whether to show the diarization result in gui
parser.add_argument('--shallShow', default=False, type=commonFunc.str2bool)
# decide whether to output the diarization result
parser.add_argument('--shallOutput', default=True, type=commonFunc.str2bool)
# decide whether load txt file to compute accept rate
parser.add_argument('--shallComputeAcc', default=False, type=commonFunc.str2bool)

global args
args = parser.parse_args()

_padTime = 150  # extends time(millisecond) for split wavs


def append2dict(speakerSlice, spk_period):
    speakerId = spk_period[0]
    timePair = spk_period[1]  # start time, end time

    timeDict = {}
    timeDict['start'] = int(timePair[0] + 0.5)  # rounded
    timeDict['stop'] = int(timePair[1] + 0.5)

    if (speakerId in speakerSlice):
        speakerSlice[speakerId].append(timeDict)
    else:
        speakerSlice[speakerId] = [timeDict]


def arrangeResult(labels, time_spec_rate):  # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
    lastLabel = labels[0]
    speakerSlice = {}
    st = 0
    for i, label in enumerate(labels):
        if label == lastLabel:
            continue
        append2dict(speakerSlice, [lastLabel, (time_spec_rate * st, time_spec_rate * i)])
        st = i
        lastLabel = label
    append2dict(speakerSlice, [lastLabel, (time_spec_rate * st, time_spec_rate * (len(labels)))])
    return speakerSlice


def genMap(intervalsAsMs, isWithoutSilence=False):  # interval slices to maptable
    key2oriFrame = {}
    if isWithoutSilence:

        pass
    else:
        slicelen = [sliced[1] - sliced[0] for sliced in intervalsAsMs.tolist()]
        # mapTable vad erased time to origin time, only split points
        idx = 0
        for i, sliced in enumerate(intervalsAsMs.tolist()):
            key2oriFrame[idx] = sliced[0]
            idx += slicelen[i]
        key2oriFrame[sum(slicelen)] = intervalsAsMs[-1, -1]

    keyFrames = [k for k, _ in key2oriFrame.items()]
    keyFrames.sort()
    return key2oriFrame, keyFrames


def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond % 1000
    minute = timeInMillisecond // 1000 // 60
    second = (timeInMillisecond - minute * 60 * 1000) // 1000
    if minute < 0 or millisecond < 0 or second < 0:
        minute = millisecond = second = 0
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time


def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    return wav, np.array(wav_output), intervals


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    return linear.T


# spec_len
# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
# |--------|
# spec_hop_len
def load_data(path, win_length, sr, hop_length, n_fft, embedding_per_second, overlap_rate):
    oriWavData, intervalWav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(intervalWav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape

    spec_len = sr / hop_length / embedding_per_second
    spec_hop_len = spec_len * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while True:  # slide window.
        if cur_slide + spec_len > time:
            break
        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_len + 0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return oriWavData, utterances_spec, intervals


def main(wavPath, embedding_per_second, num_speaker, shallOutput=False, shallShow=True):
    # check
    assert str.lower(wavPath).endswith('.wav')

    # gpu configuration
    toolkits.initialize_GPU(args)

    # params
    params = {'dim': (257, None, 1),  # 'dim': (257, None, 1),
              'sr': 16000,  # if None as original sampleRate
              'n_fft': 512,
              'win_length': 512,  # 400
              'hop_length': 128,  # 160
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    overlap_rate = (params['win_length'] - params['hop_length']) / params['win_length']

    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'], num_class=params['n_classes'], mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)

    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512

    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(args.modelPath)

    interList = None
    accRate = None

    if args.shallComputeAcc:
        interList = []
        accRate = 0.0

        fileTxt = open(wavPath[0: -len('.wav')] + ".txt", mode='r')
        for line in fileTxt.readlines():
            if not line or line == '':
                continue
            interList.append([int(intStr) for intStr in line.split(' ')])

    oriWavData, specs, intervals = load_data(wavPath, win_length=params['win_length'], sr=params['sr'], hop_length=params['hop_length'], n_fft=params['n_fft'], embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    intervalsAsMs = (intervals / params['sr'] * 1000).astype(int)
    key2oriFrame, keyFrames = genMap(intervalsAsMs)  # keys' unit is ms

    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats.append(v)

    feats = np.array(feats)[:, 0, :].astype(float)  # [splits, embedding dim]

    time_spec_rate = 1000 * (1.0 / embedding_per_second) * (1.0 - overlap_rate)  # speaker embedding every ?ms

    # bencq

    inference_args.num_speaker = num_speaker
    predicted_label = uisrnnModel.predict(feats, inference_args)

    # bencq

    speakerSlice = arrangeResult(predicted_label, time_spec_rate)

    for spkInd, timeDicts in speakerSlice.items():  # time map to orgin wav(contains mute)
        for tid, timeDict in enumerate(timeDicts):
            startTime = 0
            endTime = 0
            for i, key in enumerate(keyFrames):
                if (startTime != 0 and endTime != 0):
                    break
                if (startTime == 0 and key > timeDict['start']):
                    offset = timeDict['start'] - keyFrames[i - 1]
                    startTime = key2oriFrame[keyFrames[i - 1]] + offset
                if (endTime == 0 and key > timeDict['stop']):
                    offset = timeDict['stop'] - keyFrames[i - 1]
                    endTime = key2oriFrame[keyFrames[i - 1]] + offset

            speakerSlice[spkInd][tid]['start'] = startTime
            speakerSlice[spkInd][tid]['stop'] = endTime

    _marginTime = 1  # 1 second
    accumulateFrameCnt = [0] * 2  # classify to 0 or 1 -> 00 + 11 or 01 + 10
    _totalFrame = intervals[-1][-1] - intervals[0][0]
    for spkInd, timeDicts in speakerSlice.items():
        print('========= ' + str(spkInd) + ' =========')

        # bencq

        listIntervalWavData = []  # for output wav

        # bencq

        for timeDict in timeDicts:

            startTime = timeDict['start'] - _padTime
            endTime = timeDict['stop'] + _padTime
            sStr = fmtTime(startTime)  # change point moves to the center of the slice
            eStr = fmtTime(endTime)

            print(sStr + ' ==> ' + eStr)

            startFrame = max(0, startTime * params['sr'] // 1000)
            endFrame = min(_totalFrame, endTime * params['sr'] // 1000)

            if args.shallComputeAcc:
                # compute overlap frame count
                for inter in interList:

                    ssFrame = max(startFrame, inter[0])
                    eeFrame = min(endFrame, inter[1])

                    ind = inter[2] ^ spkInd
                    if ssFrame <= eeFrame:
                        accumulateFrameCnt[ind] += (eeFrame - ssFrame)

            if shallOutput:
                listIntervalWavData.append(oriWavData[startFrame: endFrame])
                listIntervalWavData.append(np.zeros([_marginTime * params['sr']], dtype=np.float))  # margin silence wav

        if shallOutput:
            dir, fileName = os.path.split(wavPath)
            fileNamePrefix, fileNameExtend = os.path.splitext(fileName)
            outPath = os.path.join(dir, fileNamePrefix + "_" + str(spkInd) + fileNameExtend)

            outWavData = np.concatenate(listIntervalWavData)

            soundfile.write(outPath, data=outWavData, samplerate=params['sr'], format='WAV')
            # librosa.output.write_wav(outPath, outWavData, params['sr'])

    if args.shallComputeAcc:
        accRate = min(1.0, max(accumulateFrameCnt[0], accumulateFrameCnt[1]) / _totalFrame)

    if args.shallComputeAcc:
        print('acc: %.2f%%' % (100 * accRate,))

    if shallShow:
        plotDiar = PlotDiar(map=speakerSlice, wav=wavPath, gui=True, size=(25, 6))
        plotDiar.draw()
        plotDiar.plot.show()


if __name__ == '__main__':
    main(args.wavPath, embedding_per_second=1.2, num_speaker=2, shallOutput=args.shallOutput, shallShow=args.shallShow)
