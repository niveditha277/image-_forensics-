import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten,GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from keras.applications import MobileNetV2

# Non-Binary Image Classification using Convolution Neural Networks

path = 'Dataset'

labels = []
X_train = []
Y_train = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            gray= cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create() #creating SIFT object
            step_size = 5
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                  for x in range(0, gray.shape[1], step_size)] #creating key points for SIFT to extract global features
            img = cv2.drawKeypoints(gray,kp, img)#drawing keypoints on image to extract SIFT data
            if img is not None:
                img = img.ravel()
                img = cv2.resize(img, (32,32))
                X_train.append(img.ravel())
                lbl = getID(name)
                Y_train.append(lbl)
                print(name+" "+root+"/"+directory[j]+" "+str(img.shape)+" "+str(lbl))
       
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)
np.save('model/sift_X',X_train)
np.save('model/sift_Y',Y_train)
'''
X_train = np.load('model/X.txt.npy')
Y_train = np.load('model/Y.txt.npy')

X_train = X_train.astype('float32')
X_train = X_train/255
    
test = X_train[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
Y_train = to_categorical(Y_train)



print(Y_train)

if os.path.exists('model/squeezenet_model.json'):
    with open('model/squeezenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        squeezenet = model_from_json(loaded_model_json)
    json_file.close()    
    squeezenet.load_weights("model/squeezenet_weights.h5")
    squeezenet._make_predict_function()       
else:    
    squeezenet = Sequential()
    squeezenet.add(Convolution2D(filters=6, kernel_size=3, padding='same', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    squeezenet.add(BatchNormalization())
    squeezenet.add(Activation('relu'))
    squeezenet.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    squeezenet.add(Convolution2D(filters=16, strides=1, kernel_size=5))
    squeezenet.add(BatchNormalization())
    squeezenet.add(Activation('relu'))
    squeezenet.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    squeezenet.add(GlobalAveragePooling2D())
    squeezenet.add(Dense(64, activation='relu'))
    squeezenet.add(Dense(32, activation='relu'))
    squeezenet.add(Dense(Y_train.shape[1], activation = 'softmax'))
    squeezenet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = squeezenet.fit(X_train, Y_train, batch_size=16, epochs=30, shuffle=True, verbose=2)
    squeezenet.save_weights('model/squeezenet_weights.h5')            
    model_json = squeezenet.to_json()
    with open("model/squeezenet_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/squeezenet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/squeezenet_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))
'''

'''

if os.path.exists('model/shufflenet_model.json'):
    with open('model/shufflenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        shufflenet = model_from_json(loaded_model_json)
    json_file.close()    
    shufflenet.load_weights("model/shufflenet_weights.h5")
    shufflenet._make_predict_function()       
else:    
    shufflenet = Sequential()
    shufflenet.add(Convolution2D(32, 3, 3, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    shufflenet.add(MaxPooling2D(pool_size = (2, 2)))
    shufflenet.add(Convolution2D(32, 3, 3, activation = 'relu'))
    shufflenet.add(MaxPooling2D(pool_size = (2, 2)))
    shufflenet.add(Flatten())
    shufflenet.add(Dense(output_dim = 256, activation = 'relu'))
    shufflenet.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))
    print(shufflenet.summary())
    shufflenet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = shufflenet.fit(X_train, Y_train, batch_size=16, epochs=30, shuffle=True, verbose=2)
    shufflenet.save_weights('model/shufflenet_weights.h5')            
    model_json = shufflenet.to_json()
    with open("model/shufflenet_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/shufflenet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/shufflenet_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))



if os.path.exists('model/mobilenet_model.json'):
    with open('model/mobilenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        mobilenet = model_from_json(loaded_model_json)
    json_file.close()    
    mobilenet.load_weights("model/mobilenet_weights.h5")
    mobilenet._make_predict_function()       
else:   
    mn = MobileNetV2(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    mn.trainable = False
    mobilenet = Sequential()
    mobilenet.add(mn)
    mobilenet.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    mobilenet.add(MaxPooling2D(pool_size = (1, 1)))
    mobilenet.add(Convolution2D(32, 1, 1, activation = 'relu'))
    mobilenet.add(MaxPooling2D(pool_size = (1, 1)))
    mobilenet.add(Flatten())
    mobilenet.add(Dense(output_dim = 256, activation = 'relu'))
    mobilenet.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))
    print(mobilenet.summary())
    mobilenet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = mobilenet.fit(X_train, Y_train, batch_size=16, epochs=30, shuffle=True, verbose=2)
    mobilenet.save_weights('model/mobilenet_weights.h5')            
    model_json = mobilenet.to_json()
    with open("model/mobilenet_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/mobilenet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/mobilenet_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))

'''






