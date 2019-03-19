import keras
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM , Input , Dense , Dropout
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix       
def lstm():
    input_shape = (40,12)
    model = Sequential()
    model.add(LSTM(40,return_sequences = False , input_shape = input_shape,dropout = 0.5))
    #model.add(Dense(40,activation = "relu"))
    model.add(Dense(20,activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation = "softmax"))

    model.compile(loss = "binary_crossentropy",optimizer = Adam(),metrics=['accuracy'])
    #print(model.summary())
    return model

model = lstm()

X_train = np.load("./X_train.npy")
y_train = np.load("./y_train.npy")
X_test = np.load("./X_test.npy")
y_test = np.load("./y_test.npy")

X_train = np.asarray(X_train,dtype= np.float32)
X_test = np.asarray(X_test,dtype= np.float32)
y_train = np.asarray(y_train,dtype= np.float32)
y_test = np.asarray(y_test,dtype= np.float32)

#print(X_test[0])

#print("\n\n\n")
#print(X_test[1])


model.fit(X_train,y_train,batch_size = 128,epochs=2000,shuffle=True,verbose=1,validation_data=(X_test,y_test))

model.save("./first_model.h5",overwrite=True)
#score = model.evaluate(X_train,y_train,verbose = 1)

#score_1 = model.evaluate(X_test,y_test,verbose = 1)

#print(model.predict(X_test))

#print(score)
#print(score_1)

#y_pred = model.predict(X_test)

#np.save("y_pred.npy",y_pred)
#confusion_matrix(y_test,y_pred)


#acc = np.zeros((36,))

#print(x)
#print(x.shape)

#print(score)


