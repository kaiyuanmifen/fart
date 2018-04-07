import keras


from keras import regularizers

l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))

dpt_model.add(layers.Dropout(0.5))

l2_model.add(layers.Dense(1, activation='sigmoid'))

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['acc'])


l2_model_hist = l2_model.fit(x_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test, y_test))