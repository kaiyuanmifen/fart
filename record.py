import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import webbrowser
import os
# from Preprocessing_Resampling import PredictFart
import soundfile as sf



import keras
from Preprocessing_Resampling import extractFeature, extractFeatureMemory

import numpy as np

from keras.models import load_model




def PredictFart(AudioFilleAddress,ResampleRate=3000):
    Fart = extractFeature(AudioFilleAddress,ResampleRate).T

    model=load_model('Fart_Resampling_model.h5')
    Predict=model.predict(Fart)
    np.max(Predict)

    Ratios=float(Predict[np.where(Predict>0.92)].shape[0])/ float(Predict.shape[0])
    #np.median(Predict)
    if Ratios>0.9:
        return True
    if Ratios<=0.9:
        return False



fs=44100
duration = 3
while True:
  text = raw_input("Press [Enter] to record:")
  myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float64')
  # with open('sample.wav', 'wb') as fout:
  #   fout.write(myrecording)
  print "Recording Audio"

  sd.wait()
  print "Audio recording complete , Play Audio"
  sd.play(myrecording, fs)
  sd.wait()

  # Prediciton logic here
  sf.write('sample.wav', myrecording, 44100)
  f = PredictFart('sample.wav')
  print(f)
  if f == True:
    os.system("open /Users/yuchen_zhang/workspace/build/tf/hackathon/static/fart.html")
  else:
    os.system("open /Users/yuchen_zhang/workspace/build/tf/hackathon/static/nofart.html")