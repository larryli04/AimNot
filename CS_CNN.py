# Training code

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

import cv2
import numpy as np
import random

from tqdm import tqdm

training_set_path = "CS_Data/Train/frame"
test_set_path = "CS_Data/Test"

pos_set = []
neg_set = []

tr_set = []
val_set = []

pos_tr = []
neg_tr = []
print("Converting training images to a numpy array:")

for x in tqdm(range(0, 4648)):
    neg_set.append(cv2.imread(training_set_path + str(x) + ".jpg"))
for x in tqdm(range(4648, 5126)):
    pos_set.append(cv2.imread(training_set_path + str(x) + ".jpg"))
random.shuffle(neg_set)
random.shuffle(pos_set)
neg_set = np.asarray(neg_set)
pos_set = np.asarray(pos_set)

print(neg_set.shape, pos_set.shape)

val_set = np.append(neg_set[:1000], pos_set[:50], axis=0)
tr_set = np.append(neg_set[1000:], pos_set[50:], axis=0)

tr_label = np.append(np.zeros(3648), np.ones(428))
val_label = np.append(np.zeros(1000), np.ones(50))

print(neg_set.shape, pos_set.shape, val_set.shape, tr_set.shape, tr_label.shape, val_label.shape)

#tr_set = np.asarray(tr_set)
#print(tr_set.shape)

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(74, 74, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])


model.summary()

history = model.fit(tr_set, tr_label, validation_data=(val_set, val_label), epochs=15)

y = model.predict(tr_set[-3:])
print(tr_set[-3:].shape)
print(model.predict(np.array([tr_set[3]])).round(2).shape)
print(y.round(2))

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
model.save("cs_aimv2.h5")