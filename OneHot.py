



import keras
from  Preprocessing import extractFeature

import numpy as np


#training and validation
Fart=extractFeature('./data/fart_clean1_mono.wav')
NoFart=extractFeature('./data/negative_sample_party_mono.wav')
NoFart2=extractFeature('./data/negative_sample_voice_mono.wav')

Fart.shape
NoFart.shape

#X=np.concatenate([Fart,NoFart,NoFart2],axis=1)
#Y=np.asarray([1  for i in range(Fart.shape[1])]+\
 #       [0  for i in range(NoFart.shape[1]+NoFart2.shape[1])]).astype('float32')



def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels.astype('int')):
        results[i, label] = 1.
    return results

X=np.concatenate([Fart,NoFart],axis=1)
Y=np.asarray([1  for i in range(Fart.shape[1])]+\
        [0  for i in range(NoFart.shape[1])]).astype('float32')

VecNP=np.vstack([X,Y]).T
VecNP.shape
np.random.shuffle(VecNP)
X=VecNP[:,range(40)]
Y=VecNP[:,40]

TrainSize=60000
ValidationSize=X.shape[0]-TrainSize



X_train=X[range(TrainSize),]
Y_train=Y[range(TrainSize),]
Y_train=to_one_hot(Y_train,2)

X_validation=X[range(TrainSize,TrainSize+ValidationSize),]
Y_validation=Y[range(TrainSize,TrainSize+ValidationSize),]
Y_validation=to_one_hot(Y_validation,2)

print "train size: " + str(TrainSize) + " Validation size:" + str(ValidationSize)

#Extra testing set s
NoFart3=extractFeature('./data/test_negative.wav').T

LzFart=np.vstack([extractFeature('./data/test_positive.wav').T,
                extractFeature('./data/fart1.wav').T,
                  extractFeature('./data/fart2.wav').T,
                  extractFeature('./data/fart3.wav').T,
                  extractFeature('./data/fart4.wav').T,
                  extractFeature('./data/fart5.wav').T,
                  extractFeature('./data/fart6.wav').T,
                  extractFeature('./data/Recording.wav').T])


X_Test=np.vstack([LzFart,NoFart3])

Y_Test=np.asarray([1 for i in range(LzFart.shape[0])]+
                  [0 for i in range(NoFart3.shape[0])]).astype('float32')

Y_Test=to_one_hot(Y_Test,2)
X_Test.shape
Y_Test.shape

#model
from keras import regularizers
from keras import models
from keras import layers


l2_model = models.Sequential()
l2_model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(40,)))
l2_model.add(layers.Dropout(0.5))
l2_model.add(layers.Dense(2, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))

#l2_model.add(layers.Dropout(0.5))

#l2_model.add(layers.Dense(20, kernel_regularizer=regularizers.l2(0.001),
                          #activation='relu'))

#l2_model.add(layers.Dropout(0.5))

l2_model.add(layers.Dense(2, activation='softmax'))

l2_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
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

Predict=l2_model.predict(X_validation)
type(Predict)
Real=Y_validation
np.corrcoef(Predict[:,1],Real[:,1])

#np.corrcoef(Predict,Real)

plt.clf()   # clear figure
plt.plot(Predict,Real,'bo')


plt.show()



#new test
from keras.models import load_model

from itertools import chain

#model2=load_model('Fart_model.h5')

l2_model.evaluate(x=X_Test, y = Y_Test, batch_size=None, verbose=1)

#Predict=list(chain.from_iterable(l2_model.predict(X_Test,).tolist()))
Predict=l2_model.predict(X_Test)
type(Predict)
Real=Y_Test
np.corrcoef(Predict[:,0],Real[:,0])

plt.clf()   # clear figure
plt.plot(Predict[:,1],Real[:,1],'bo')

plt.show()

