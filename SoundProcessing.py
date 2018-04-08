import librosa
data, sampling_rate = librosa.load('/Users/dianboliu/Downloads/fart_clean1_mono.wav')

#Resample the data ?
data, sampling_rate = librosa.load('/Users/dianboliu/Downloads/fart_clean1_mono.wav')


import os
import pandas as pd
import librosa
import glob

import matplotlib.pyplot as plt
import numpy as np

data.shape
sampling_rate



mfccs=np.mean(librosa.feature.mfcc(data,sr=sampling_rate,n_mfcc=40).T,axis=0)
mfccs.shape

X=mfccs.tolist()


def parser(row):
    # function to load files and extract features
    file_name = os.path.join(os.path.abspath(data_dir), 'Train', str(row.ID) + '.wav')

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    feature = mfccs
    label = row.Class

    return [feature, label]


temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']