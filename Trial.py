import keras
from  Preprocessing import extractFeature

import numpy as np


#training and validation
Fart=extractFeature('./data/fart_clean1_mono.wav')
NoFart=extractFeature('./data/negative_sample_party_mono.wav')
NoFart2=extractFeature('./data/negative_sample_voice_mono.wav')

X=np.concatenate([Fart,NoFart,NoFart2],axis=1)
Y=np.asarray([1  for i in range(Fart.shape[1])]+\
        [0  for i in range(NoFart.shape[1]+NoFart2.shape[1])]).astype('float32')

VecNP=np.vstack([X,Y]).T
VecNP.shape
np.random.shuffle(VecNP)
X=VecNP[:,range(40)]
Y=VecNP[:,40]

TrainSize=150000
ValidationSize=X.shape[0]-TrainSize



X_train=X[range(TrainSize),]
Y_train=Y[range(TrainSize),]

X_validation=X[range(TrainSize,TrainSize+ValidationSize),]
Y_validation=Y[range(TrainSize,TrainSize+ValidationSize),]

print "train size: " + str(TrainSize) + " Validation size:" + str(ValidationSize)

#Extra testing set s
X_Fart3=extractFeature('./data/test_positive.wav')
X_NoFart3=extractFeature('./data/test_negative.wav')

X_Test2=np.concatenate([X_Fart3,X_NoFart3],axis=1).T

Y_Fart3=np.asarray([1 for i in range(X_Fart3.shape[1])]).astype('float32')
Y_NoFart3=np.asarray([0 for i in range(X_NoFart3.shape[1])]).astype('float32')
Y_Test2=np.concatenate([Y_Fart3,Y_NoFart3],axis=0)
Y_Test2.shape



#model
from keras import regularizers
from keras import models
from keras import layers


l2_model = models.Sequential()
l2_model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(40,)))
l2_model.add(layers.Dropout(0.5))
l2_model.add(layers.Dense(5, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))

#l2_model.add(layers.Dropout(0.5))

#l2_model.add(layers.Dense(20, kernel_regularizer=regularizers.l2(0.001),
                          #activation='relu'))

#l2_model.add(layers.Dropout(0.5))

l2_model.add(layers.Dense(1, activation='sigmoid'))

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['acc'])


l2_model_hist = l2_model.fit(X_train, Y_train,
                             epochs=10,
                             batch_size=32,
                             validation_data=(X_validation, Y_validation))

#l2_model.save('Fart_model.h5')



import matplotlib.pyplot as plt
l2_model_hist
acc = l2_model_hist.history['acc']
val_acc = l2_model_hist.history['val_acc']
loss = l2_model_hist.history['loss']
val_loss = l2_model_hist.history['val_loss']

epochs = range(1, len(acc) + 1)

# # "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.plot(epochs, acc, 'bo', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.show()





#new test
from keras.models import load_model

from itertools import chain

#model2=load_model('Fart_model.h5')

l2_model.evaluate(x=X_validation, y = Y_validation, batch_size=None, verbose=1)

Predict=list(chain.from_iterable(l2_model.predict(X_Test2).tolist()))
type(Predict)
Real=Y_Test2.tolist()
np.corrcoef(Predict,Real)

plt.clf()   # clear figure
plt.plot(Predict,Real,'bo')

plt.show()