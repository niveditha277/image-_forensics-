from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import cv2
from tensorflow.keras.models import model_from_json, Model
from sklearn import svm
import pandas as pd

# Initialize main tkinter window
main = Tk()
main.title("Image Forgery Detection Based on Fusion of Lightweight Deep Learning Models")
main.geometry("1200x1200")

# Global Variables
global X_train, X_test, y_train, y_test, fine_features
global filename, X, Y
accuracy, precision, recall, fscore = [], [], [], []
squeezenet, shufflenet, mobilenet = None, None, None
labels = ['Non Forged', 'Forged']

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=f"{filename} Dataset Loaded")
    text.insert(END, f"{filename} Dataset Loaded\n\n")

def preprocessDataset():
    global X, Y
    text.delete('1.0', END)
    try:
        X, Y = np.load('model/sift_X.npy'), np.load('model/sift_Y.npy')
        X = X.astype('float32') / 255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, Y = X[indices], Y[indices]
        text.insert(END, f"Total images found in dataset: {X.shape[0]}\n\n")
        test_img = cv2.resize(X[10], (100, 100))
        cv2.imshow("Sample Processed Image", test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        text.insert(END, f"Error loading dataset: {str(e)}\n")

def getMetrics(predictions, actual, model_name):
    # Convert actual to single labels if it's one-hot encoded
    if actual.ndim > 1:
        actual = np.argmax(actual, axis=1)
    
    p = precision_score(actual, predictions, average='macro', zero_division=0) * 100
    r = recall_score(actual, predictions, average='macro', zero_division=0) * 100
    f = f1_score(actual, predictions, average='macro', zero_division=0) * 100
    a = accuracy_score(actual, predictions) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, f"{model_name} Precision: {p}\n{model_name} Recall: {r}\n{model_name} F1 Score: {f}\n{model_name} Accuracy: {a}\n\n")

def loadModel(model_name):
    try:
        with open(f'model/{model_name}_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(f"model/{model_name}_weights.h5")
        return model
    except Exception as e:
        text.insert(END, f"Error loading model {model_name}: {str(e)}\n")
        return None

def fusionModel():
    global X_train, X_test, y_train, y_test, fine_features, squeezenet, shufflenet, mobilenet
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    squeezenet = loadModel("squeezenet")
    if squeezenet:
        predict = np.argmax(squeezenet.predict(X_test), axis=1)
        getMetrics(predict, y_test, "SqueezeNet")
    
    shufflenet = loadModel("shufflenet")
    if shufflenet:
        predict = np.argmax(shufflenet.predict(X_test), axis=1)
        getMetrics(predict, y_test, "ShuffleNet")
    
    mobilenet = loadModel("mobilenet")
    if mobilenet:
        predict = np.argmax(mobilenet.predict(X_test), axis=1)
        getMetrics(predict, y_test, "MobileNetV2")
    
    squeeze_features = Model(squeezenet.inputs, squeezenet.layers[-3].output).predict(X)
    shuffle_features = Model(shufflenet.inputs, shufflenet.layers[-3].output).predict(X)
    mobile_features = Model(mobilenet.inputs, mobilenet.layers[-3].output).predict(X)
    
    fine_features = np.column_stack((squeeze_features, shuffle_features, mobile_features))
    text.insert(END, f"Total fine-tuned features extracted: {fine_features.shape[1]}\n\n")

def finetuneSVM():
    global X_train, X_test, y_train, y_test
    svm_cls = svm.SVC()
    
    # Ensure y_train is 1D
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    
    svm_cls.fit(fine_features[:X_train.shape[0]], y_train)  # Use y_train for fitting
    predict = svm_cls.predict(fine_features[X_train.shape[0]:])  # Predict on the test set
    
    # Ensure y_test is also in the correct format
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    
    getMetrics(predict, y_test, "Fusion Model SVM")
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, predict), xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    plt.title("Fusion Model Confusion Matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def graph():
    metrics = ["Precision", "Recall", "F1 Score", "Accuracy"]
    models = ["SqueezeNet", "ShuffleNet", "MobileNetV2", "Fusion Model SVM"]
    
    # Ensure all metrics lists have the same length
    if not (len(precision) == len(recall) == len(fscore) == len(accuracy)):
        text.insert(END, "Error: Metrics lists have mismatched lengths\n")
        return  
    
    df = pd.DataFrame({
        "Model": models * 4,  # Repeat each model name 4 times for each metric
        "Metric": metrics * len(models),
        "Value": precision[:4] + recall[:4] + fscore[:4] + accuracy[:4]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Metric", y="Value", hue="Model", data=df, palette="coolwarm")
    plt.title("Performance Comparison of Different Models")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Evaluation Metrics")
    plt.ylim(0, 100)
    plt.legend(title="Models")
    plt.show()

def close():
    main.destroy()

# UI Elements
font = ('times', 14, 'bold')
title = Label(main, text='Image Forgery Detection Based on Fusion of Lightweight Deep Learning Models', bg='DarkGoldenrod1', fg='black', font=font, height=3, width=120)
title.place(x=5, y=5)

buttons = [
    ("Upload Dataset", uploadDataset),
    ("Preprocess Dataset", preprocessDataset),
    ("Generate & Load Fusion Model", fusionModel),
    ("Fine Tune SVM", finetuneSVM),
    ("Show Performance Graph", graph),
    ("Exit", close)
]

for i, (text, command) in enumerate(buttons):
    Button(main, text=text, command=command, font=('times', 13, 'bold')).place(x=50, y=100 + (i * 50))

pathlabel = Label(main, bg='brown', fg='white', font=('times', 13, 'bold'))
pathlabel.place(x=400, y=100)

text = Text(main, height=25, width=100, font=('times', 12, 'bold'))
text.place(x=400, y=150)

main.config(bg='LightSteelBlue1')
main.mainloop()