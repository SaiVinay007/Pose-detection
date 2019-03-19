import keras
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM , Input , Dense , Dropout
from keras.optimizers import Adam

def lstm():
    input_shape = (40,12)
    model = Sequential()
    model.add(LSTM(40,return_sequences = False , input_shape = input_shape,dropout = 0.5))
    model.add(Dense(2,activation = "softmax"))
    model.compile(loss = "binary_crossentropy",optimizer = Adam)
    #print(model.summary())
    return model

model = lstm()


model.fit(x_train,y_train)

#model.save_weights("./first_model.h5",overwrite=True)







  
