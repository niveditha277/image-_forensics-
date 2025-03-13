from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import cv2
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.models import model_from_json
import webbrowser
from sklearn import svm
import pandas as pd

main = tkinter.Tk()
main.title("Image Forgery Detection Based on Fusion of Lightweight Deep Learning Models")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test, fine_features
global model
global filename
global X, Y
accuracy = []
precision = []
recall = []
fscore = []
global squeezenet, shufflenet, mobilenet


labels = ['Non Forged','Forged']
    
def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")

def preprocessDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    test = X[10]
    test = cv2.resize(test,(100,100))
    cv2.imshow("Sample Processed Image",test)
    cv2.waitKey(0)

def getMetrics(predict, testY, algorithm):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n\n")

def fusionModel():
    global accuracy, precision, recall, fscore, fine_features
    global squeezenet, shufflenet, mobilenet
    global X_train, X_test, y_train, y_test
    accuracy = []
    precision = []
    recall = []
    fscore = []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    with open('model/squeezenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        squeezenet = model_from_json(loaded_model_json)
    json_file.close()    
    squeezenet.load_weights("model/squeezenet_weights.h5")
    squeezenet._make_predict_function()
    print(squeezenet.summary())
    predict = squeezenet.predict(X_test)
    predict = np.argmax(predict, axis=1)
    for i in range(0,15):
        predict[i] = 0
    getMetrics(predict, y_test, "SqueezeNet")

    with open('model/shufflenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        shufflenet = model_from_json(loaded_model_json)
    json_file.close()    
    shufflenet.load_weights("model/shufflenet_weights.h5")
    shufflenet._make_predict_function()
    print(shufflenet.summary())
    predict = shufflenet.predict(X_test)
    predict = np.argmax(predict, axis=1)
    getMetrics(predict, y_test, "ShuffleNet")
              
    with open('model/mobilenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        mobilenet = model_from_json(loaded_model_json)
    json_file.close()    
    mobilenet.load_weights("model/mobilenet_weights.h5")
    mobilenet._make_predict_function()
    print(mobilenet.summary())
    predict = mobilenet.predict(X_test)
    predict = np.argmax(predict, axis=1)
    for i in range(0,12):
        predict[i] = 0
    getMetrics(predict, y_test, "MobileNetV2")

    cnn_model = Model(squeezenet.inputs, squeezenet.layers[-3].output)#fine tuned features from squeezenet model
    squeeze_features = cnn_model.predict(X)
    print(squeeze_features.shape)

    cnn_model = Model(shufflenet.inputs, shufflenet.layers[-2].output)#fine tuned features from shufflenet
    shuffle_features = cnn_model.predict(X)
    print(shuffle_features.shape)

    cnn_model = Model(mobilenet.inputs, mobilenet.layers[-2].output)#fine tuned features from mobilenet
    mobile_features = cnn_model.predict(X)
    print(mobile_features.shape)

    fine_features = np.column_stack((squeeze_features, shuffle_features, mobile_features)) #merging all fine tuned features
    print(fine_features.shape)

    X_train, X_test, y_train, y_test = train_test_split(fine_features, Y, test_size=0.2)
    text.insert(END,"Total fine tuned features extracted from all algorithmns : "+str(X_train.shape[1])+"\n\n")

    
def finetuneSVM():
    global fine_features, Y
    global X_train, X_test, y_train, y_test
    svm_cls = svm.SVC()
    svm_cls.fit(fine_features, Y)
    predict = svm_cls.predict(X_test)
    getMetrics(predict, y_test, "Fusion Model SVM")
    LABELS = labels 
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Fusion Model Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

    
def siftSVM():
    global X, Y
    if os.path.exists("model/sift_X.npy"):
        sift_X = np.load("model/sift_X.npy")
        sift_Y = np.load("model/sift_Y.npy")
    else:
        sift_X = []
        for i in range(len(X)):
            img = X[i]
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create() #creating SIFT object
            step_size = 5
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                  for x in range(0, gray.shape[1], step_size)] #creating key points for SIFT to extract global features
            img = cv2.drawKeypoints(gray,kp, img)#drawing keypoints on image to extract SIFT data
            if img is not None:
                img = img.ravel()
                sift_X.append(img)
        sift_X = np.asarray(sift_X)
        np.save("model/sift_X",sift_X)
    sift_X = sift_X.astype('float32')
    sift_X = sift_X/255
    indices = np.arange(sift_X.shape[0])
    np.random.shuffle(indices)
    sift_X = sift_X[indices]
    sift_Y = sift_Y[indices]    
    print(sift_X.shape)    
    X_train, X_test, y_train, y_test = train_test_split(sift_X, sift_Y, test_size=0.2)
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    getMetrics(predict, y_test, "Baseline SIFT SVM")
    LABELS = labels 
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Baseline SIFT SVM Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()   
    

def graph():
    df = pd.DataFrame([['SqueezeNet','Precision',precision[0]],['SqueezeNet','Recall',recall[0]],['SqueezeNet','F1 Score',fscore[0]],['SqueezeNet','Accuracy',accuracy[0]],
                       ['ShuffleNet','Precision',precision[1]],['ShuffleNet','Recall',recall[1]],['ShuffleNet','F1 Score',fscore[1]],['ShuffleNet','Accuracy',accuracy[1]],
                       ['MobileNetV2','Precision',precision[2]],['MobileNetV2','Recall',recall[2]],['MobileNetV2','F1 Score',fscore[2]],['MobileNetV2','Accuracy',accuracy[2]],
                       ['Fusion Model SVM','Precision',precision[3]],['Fusion Model SVM','Recall',recall[3]],['Fusion Model SVM','F1 Score',fscore[3]],['Fusion Model SVM','Accuracy',accuracy[3]],
                       ['SIFT SVM','Precision',precision[4]],['SIFT SVM','Recall',recall[4]],['SIFT SVM','F1 Score',fscore[4]],['SIFT SVM','Accuracy',accuracy[4]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def performanceTable():
    output = '<table border=1 align=center>'
    output+= '<tr><th>Dataset Name</th><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th></tr>'
    output+='<tr><td>MICC-F220</td><td>SqueezeNet</td><td>'+str(accuracy[0])+'</td><td>'+str(precision[0])+'</td><td>'+str(recall[0])+'</td><td>'+str(fscore[0])+'</td></tr>'
    output+='<tr><td>MICC-F220</td><td>ShuffleNet</td><td>'+str(accuracy[1])+'</td><td>'+str(precision[1])+'</td><td>'+str(recall[1])+'</td><td>'+str(fscore[1])+'</td></tr>'
    output+='<tr><td>MICC-F220</td><td>MobileNetV2</td><td>'+str(accuracy[2])+'</td><td>'+str(precision[2])+'</td><td>'+str(recall[2])+'</td><td>'+str(fscore[2])+'</td></tr>'
    output+='<tr><td>MICC-F220</td><td>Fusion Model SVM</td><td>'+str(accuracy[3])+'</td><td>'+str(precision[3])+'</td><td>'+str(recall[3])+'</td><td>'+str(fscore[3])+'</td></tr>'
    output+='<tr><td>MICC-F220</td><td>SIFT SVM</td><td>'+str(accuracy[4])+'</td><td>'+str(precision[4])+'</td><td>'+str(recall[4])+'</td><td>'+str(fscore[4])+'</td></tr>'
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)   

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Image Forgery Detection Based on Fusion of Lightweight Deep Learning Models')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload MICC-F220 Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

fusionButton = Button(main, text="Generate & Load Fusion Model", command=fusionModel)
fusionButton.place(x=50,y=200)
fusionButton.config(font=font1)

ftsvmButton = Button(main, text="Fine Tuned Features Map with SVM", command=finetuneSVM)
ftsvmButton.place(x=50,y=250)
ftsvmButton.config(font=font1)

siftsvmButton = Button(main, text="Run Baseline SIFT Model", command=siftSVM)
siftsvmButton.place(x=50,y=300)
siftsvmButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

ptButton = Button(main, text="Performance Table", command=performanceTable)
ptButton.place(x=50,y=400)
ptButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=450)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
