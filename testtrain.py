import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical  # FIXED import
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, Conv2D
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.applications import MobileNetV2

# Create necessary directories
dataset_path = 'Dataset'
model_path = 'model'
os.makedirs(model_path, exist_ok=True)  # Ensure 'model' directory exists

labels = []
X_train = []
Y_train = []

# Function to get label index
def getID(name):
    if name in labels:
        return labels.index(name)
    labels.append(name)
    return len(labels) - 1

# Load dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            name = os.path.basename(root)
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sift = cv2.SIFT_create()  # Create SIFT object
                kp = [cv2.KeyPoint(x, y, 5) for y in range(0, gray.shape[0], 5) for x in range(0, gray.shape[1], 5)]
                img = cv2.drawKeypoints(gray, kp, img)
                img = cv2.resize(img, (32, 32))
                
                X_train.append(img / 255.0)  # Normalize pixel values
                Y_train.append(getID(name))
                print(f"Processed: {name} - {img_path}")

# Convert to numpy arrays
X_train = np.array(X_train, dtype='float32')
Y_train = to_categorical(np.array(Y_train))

# Shuffle dataset
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train, Y_train = X_train[indices], Y_train[indices]

# Save preprocessed data
np.save(f'{model_path}/sift_X.npy', X_train)
np.save(f'{model_path}/sift_Y.npy', Y_train)

# Function to load or create a model
def load_or_create_model(model_name, model_fn):
    json_file_path = f"{model_path}/{model_name}_model.json"
    weights_file_path = f"{model_path}/{model_name}_weights.h5"
    history_file_path = f"{model_path}/{model_name}_history.pckl"

    if os.path.exists(json_file_path) and os.path.exists(weights_file_path):
        with open(json_file_path, "r") as json_file:
            model = model_from_json(json_file.read())
        model.load_weights(weights_file_path)
        print(f"Loaded existing {model_name} model.")
    else:
        model = model_fn()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=16, epochs=30, shuffle=True, verbose=2)

        model.save_weights(weights_file_path)
        with open(json_file_path, "w") as json_file:
            json_file.write(model.to_json())
        with open(history_file_path, 'wb') as f:
            pickle.dump(hist.history, f)

    return model

# Define models
def create_squeezenet():
    model = Sequential([
        Conv2D(6, kernel_size=3, padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(), Activation('relu'),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        Conv2D(16, kernel_size=5, activation='relu'),
        BatchNormalization(), Activation('relu'),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(Y_train.shape[1], activation='softmax')
    ])
    return model

def create_shufflenet():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(Y_train.shape[1], activation='softmax')
    ])
    return model

def create_mobilenet():
    mn = MobileNetV2(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    mn.trainable = False
    model = Sequential([
        mn,
        Conv2D(32, kernel_size=1, activation='relu'),
        MaxPooling2D(pool_size=1),
        Conv2D(32, kernel_size=1, activation='relu'),
        MaxPooling2D(pool_size=1),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(Y_train.shape[1], activation='softmax')
    ])
    return model

# Load or train models
squeezenet = load_or_create_model("squeezenet", create_squeezenet)
shufflenet = load_or_create_model("shufflenet", create_shufflenet)
mobilenet = load_or_create_model("mobilenet", create_mobilenet)

print("✅ Training complete!")
