
"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import numpy as np
import uisrnn
import librosa
import sys
sys.path.append('ghostvlad')
sys.path.append('visualization')
import toolkits
import model as spkModel
import os
from viewer import PlotDiar

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)



global args
args = parser.parse_args()


SAVED_MODEL_NAME = 'pretrained/mSave.uisrnn'
SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'

_padTime = 150  # extends time(millisecond) for split wavs

def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0]+0.5)  # rounded
    timeDict['stop'] = int(value[1]+0.5)
    if(key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice

def arrangeResult(labels, time_spec_rate): # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i,label in enumerate(labels):
        if(label==lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*(len(labels)))})
    return speakerSlice


def genMap(intervals, isWithoutSilence = False):  # interval slices to maptable
    mapTable = {}
    if isWithoutSilence:

        pass
    else:
        slicelen = [sliced[1]-sliced[0] for sliced in intervals.tolist()]
        # mapTable vad erased time to origin time, only split points
        idx = 0
        for i, sliced in enumerate(intervals.tolist()):
            mapTable[idx] = sliced[0]
            idx += slicelen[i]
        mapTable[sum(slicelen)] = intervals[-1,-1]

    keys = [k for k,_ in mapTable.items()]
    keys.sort()
    return mapTable, keys

def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond%1000
    minute = timeInMillisecond//1000//60
    second = (timeInMillisecond-minute*60*1000)//1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time

def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    return wav, np.array(wav_output), (intervals*1000/sr).astype(int)

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5, overlap_rate=0.5):
    oriWavData, intervalWav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(intervalWav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr/hop_length/embedding_per_second
    spec_hop_len = spec_len*(1-overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while(True):  # slide window.
        if(cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide+0.5) : int(cur_slide+spec_len+0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return oriWavData, utterances_spec, intervals

def main(wav_path, embedding_per_second=1.0, overlap_rate=0.5, num_speaker=0, shallOutput = False, shallShow = True):

    # gpu configuration
    toolkits.initialize_GPU(args)

    params = {'dim': (257, None, 1), # 'dim': (257, None, 1),
              'sr': 16000, # if None as original sampleRate
              'n_fft': 512,
              'win_length': 512,  # 400
              'hop_length': 128,  # 160
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)


    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512


    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)

    oriWavData, specs, intervals = load_data(wav_path, win_length=params['win_length'], sr=params['sr'], hop_length=params['hop_length'], n_fft=params['n_fft'], embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    mapTable, keys = genMap(intervals)

    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats += [v]

    feats = np.array(feats)[:,0,:].astype(float)  # [splits, embedding dim]


    # bencq

    inference_args.num_speaker = num_speaker
    predicted_label = uisrnnModel.predict(feats, inference_args)

    # bencq

    time_spec_rate = 1000*(1.0/embedding_per_second)*(1.0-overlap_rate) # speaker embedding every ?ms
    center_duration = int(1000*(1.0/embedding_per_second)//2)
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)

    for spkInd,timeDicts in speakerSlice.items():    # time map to orgin wav(contains mute)
        for tid,timeDict in enumerate(timeDicts):
            s = 0
            e = 0
            for i,key in enumerate(keys):
                if(s!=0 and e!=0):
                    break
                if(s==0 and key>timeDict['start']):
                    offset = timeDict['start'] - keys[i-1]
                    s = mapTable[keys[i-1]] + offset
                if(e==0 and key>timeDict['stop']):
                    offset = timeDict['stop'] - keys[i-1]
                    e = mapTable[keys[i-1]] + offset

            speakerSlice[spkInd][tid]['start'] = s
            speakerSlice[spkInd][tid]['stop'] = e

    _marginTime = 1  # 1sec
    for spkInd,timeDicts in speakerSlice.items():
        print('========= ' + str(spkInd) + ' =========')

        # bencq

        listIntervalWavData = []

        # bencq

        for timeDict in timeDicts:
            s = timeDict['start']
            e = timeDict['stop']
            sStr = fmtTime(s)  # change point moves to the center of the slice
            eStr = fmtTime(e)
            print(sStr + ' ==> '+ eStr)

            if shallOutput:
                listIntervalWavData.append(oriWavData[(s-_padTime)*params['sr'] // 1000: (e+_padTime)*params['sr'] // 1000])
                listIntervalWavData.append(np.zeros([ _marginTime*params['sr'] ], dtype=np.float))  # margin silence wav

        if shallOutput:
            dir, fileName = os.path.split(wav_path)
            fileNamePrefix, fileNameExtend = os.path.splitext(fileName)
            outPath = os.path.join(dir, fileNamePrefix + "_" + str(spkInd) + fileNameExtend)

            outWavData = np.concatenate(listIntervalWavData)
            librosa.output.write_wav(outPath, outWavData, params['sr'])

    if shallShow:
        # p = PlotDiar(map=speakerSlice, wav=wav_path, gui=True, size=(25, 6))
        # p.draw()
        # p.plot.show()
        p = PlotDiar(map=speakerSlice, wav=wav_path, gui=True, size=(25, 6))
        p.draw()
        p.plot.show()

if __name__ == '__main__':

    # wavPath = 'F:/tempMaterial/rec.wav'
    wavPath = r'wavs/eng_vad.wav'
    # wavPath = r'wavs/柴宋博-俊业电话录音_vad.wav'
    # wavPath = r'wavs/陈海峰-俊业电话录音_vad.wav'
    # wavPath = r'wavs/mix1_vad.wav'
    # wavPath = r'wavs/mixA3A4.wav'
    wavPath = r'wavs/mixA4A5.wav'
    # wavPath = r'wavs/mixA5L5_vad.wav'
    # wavPath = r'E:\source_code\python\
    # keras\test\venv\DeepSpeechRecognition\data\data_thchs30\data\A2_0.wav'
    # wavPath = r'wavs/LDC2005S15mix_vad.wav'
    # embedding_per_second=1.2, overlap_rate=0.5
    main(wavPath, embedding_per_second=1.2, overlap_rate=0.5, num_speaker=2, shallOutput = False, shallShow = True)

