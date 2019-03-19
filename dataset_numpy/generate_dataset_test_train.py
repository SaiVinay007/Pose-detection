import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
path = "./"

all_arrays = os.listdir(path+"Both_groups")

infile = np.zeros((len(all_arrays),40,12))
#in_final = np.zeros((None,40,12))
labels = np.zeros((len(all_arrays),))
for i, numpy_file in enumerate(all_arrays):
    if "class_1" in numpy_file:
        numpy_array = np.load(path+"Both_groups/"+numpy_file)
        if (numpy_array.shape ==(40,12)):
            
            #print("\n This is class 1 \n")
            numpy_array = (numpy_array - np.mean(numpy_array))/np.std(numpy_array)
            infile[i] = numpy_array
            labels[i] = 1
            del(numpy_array)
        #else:
        #    np.delete(infile,i,0)
        #    np.delete(labels,i)
        #    i-=1

    else:
        numpy_array = np.load(path+"Both_groups/"+numpy_file)
        if (numpy_array.shape ==(40,12)):
            numpy_array = (numpy_array - np.mean(numpy_array))/np.std(numpy_array)
            #print(np.std(numpy_array))
            if(np.isnan(np.std(numpy_array))):
                print("The numpy array is nan")
            infile[i] = numpy_array
            labels[i] = 0
            del(numpy_array)
        #else:
        #    np.delete(infile,i,0)
        #    np.delete(infile,i,0)
        #    i-=1
labels = labels.reshape(labels.shape[0],1)
count = 0
for i in range(360):
    if(np.isnan(np.std(infile[i]))):
        continue
    elif(count ==0):
        in_final = infile[i].reshape(1,40,12)
        count+=1
        label_final = labels[i]
    else:
        in_final = np.concatenate((in_final,infile[i].reshape(1,40,12)))
        #print(label_final.shape)
        #print(labels[i].shape)
        label_final = np.concatenate((label_final,labels[i]))




data,Label = shuffle(in_final,label_final)
train_data = [data,Label]
del(data,Label)
(X,y) = (train_data[0],train_data[1])
del(train_data)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
del(X,y)

y_train = np_utils.to_categorical(y_train,num_classes=2)
y_test = np_utils.to_categorical(y_test,num_classes=2)

np.save(path+"X_train.npy",X_train)
np.save(path+"X_test.npy",X_test)
np.save(path+"y_train.npy",y_train)
np.save(path+"y_test.npy",y_test)

print(X_train.shape)
print(X_test.shape)


