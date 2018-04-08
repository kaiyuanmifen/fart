



import keras
from  Preprocessing_Resampling import extractFeature

import numpy as np

from keras.models import load_model



def PredictFart(AudioFilleAddress,ResampleRate=3000):
    Fart = extractFeature(AudioFilleAddress,ResampleRate).T

    model=load_model('Fart_Resampling_model.h5')
    Predict=model.predict(Fart)
    np.max(Predict)

    Ratios=float(Predict[np.where(Predict>0.90)].shape[0])/ float(Predict.shape[0])
    #np.median(Predict)
    if Ratios>0.9:
        return True
    if Ratios<=0.9:
        return False

