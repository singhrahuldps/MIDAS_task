#! /usr/bin/env python

import sys
import librosa
from scipy.io import wavfile
import numpy as np
import pandas as pd
from fastai.script import *
from fastai.vision import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from collections import Counter

d = {'disgust':2, 'fear':4, 'happy':1, 'neutral':0, 'sad':3}
d_rev = {2:'disgust', 4:'fear', 1:'happy', 0:'neutral', 3:'sad'}

def sound2features(filename):
    sr, y = wavfile.read(filename)
    y = y.astype('float32')
    y = np.mean(y,axis=1)/32767
    mfcc = librosa.feature.mfcc(
                y=y, sr=sr, hop_length=512)
    spectral_center = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, hop_length=512)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)
    zero_crossing = librosa.feature.zero_crossing_rate(y, hop_length=512)
    #mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512)
    res = np.concatenate((mfcc,spectral_center, chroma, spectral_contrast, spectral_bandwidth, flatness, zero_crossing), 0)
    res = np.mean(res, axis=1)
    return res

def folder2features(folder):
    x = []
    fnames = []
    for soundfile in folder.ls():
        x.append(sound2features(str(soundfile)))
        fnames.append(str(soundfile).split("/")[-1])
    x = np.array(x)
    return x, fnames

@call_parse
def main(test_path):
	
	with open('xtrain.pkl','rb') as f: x_train = pickle.load(f)
	with open('ytrain.pkl','rb') as f: y_train = pickle.load(f)

	x_test, fnames = folder2features(Path(test_path))

	scaler = StandardScaler().fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	sel = SelectKBest(f_classif, k=40)
	xt = sel.fit_transform(x_train, y_train)
	xtest = sel.transform(x_test)

	clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
	clf.fit(xt, y_train)

	y_pred = clf.predict(xtest)
	sub = pd.DataFrame(columns=['File name', 'prediction'])
	sub['File name'] = fnames
	ypred = []
	for y in y_pred:
		ypred.append(d_rev[y])
	sub['prediction'] = ypred
	sub.to_csv('output.csv', index = False)