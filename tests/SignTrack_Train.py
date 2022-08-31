from calendar import EPOCH
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from keras.callbacks import History
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random

from Packr import ModelPack

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Dataset')

# Path to save model
MODEL_PATH = 'Insights/SignTrack.h5'

# Signs that we try to detect
signs = np.array(['no', 'thank you', 'me', 'please', 'good', 'morning', 'want', 'go to', 'night', 'how',
                  'hello', "what's up", 'yes', 'fine', 'see you later', 'like', 'afternoon', 'you', "sorry", 'goodbye'])

# Videos are going to be 24 frames in length
sequence_length = 24

# Creating a label map, where each sign is assigned to a specific numerical value
label_map = {label: num for num, label in enumerate(signs)}

# Importing data from dataset
sequences, labels = [], []
for sign in signs:
    print('Importing data for {}...'.format(sign))
    dirs = os.listdir('Dataset/' + sign)
    for dir in dirs:
        window = []
        window_aug = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, sign, str(
                dir), "{}.npy".format(frame_num)))
            window.append(res)
            window_aug.append(res)
        # Randomly duplicating images in a copy of res
        # Used for data augmentation
        randposs= random.randint(1, 5)
        if randposs==5:
            for i in range(round(sequence_length * random.randrange(4,6)* 0.1)):
                rand = random.randint(1, sequence_length-1)
                window_aug[rand] = window_aug[rand-1]
            sequences.append(window_aug)
            labels.append(label_map[sign])
        sequences.append(window)
        labels.append(label_map[sign])
    print('Data for {} imported \n'.format(sign))

X = np.array(sequences)
y = to_categorical(labels).astype(int)

log_dir = os.path.join('Insights')

# Splitting dataset into Train_Set and Test_set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Setting up model parameters
model = Sequential()
model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(24, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))


AutoTrain = model.compile(optimizer='Adam', loss='categorical_crossentropy')

loss = 1

AutoTrain = model.fit(X_train, y_train, epochs=1)

for i in range(250):
    if AutoTrain.history['loss'][-1] >= 0.08:
        AutoTrain = model.fit(X_train, y_train, epochs=1)
        if AutoTrain.history['loss'][-1] < loss:

            try:
                os.remove(MODEL_PATH)
            except:
                pass

            model.save(MODEL_PATH)

            loss = AutoTrain.history['loss'][-1]

ModelPack(MODEL_PATH, signs)
