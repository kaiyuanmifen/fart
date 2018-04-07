



import keras
from  Preprocessing import extractFeature

import numpy as np

from keras.models import load_model



def PredictFart(AudioFilleAddress):
    Fart = extractFeature(AudioFilleAddress).T

    model=load_model('Fart_model.h5')
    Predict=model.predict(Fart)
    #np.max(Predict)
    Ratios=float(Predict[np.where(Predict>0.5)].shape[0])/ float(Predict.shape[0])
    #np.median(Predict)
    if Ratios>0.9:
        return True
    if Ratios<=0.9:
        return False

