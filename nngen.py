import sys
import keras.layers
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, mixed_precision

import os
import numpy as np
import csv


# Load the binary-encoded labels from the CSV file
csv_file = "newDat/true_labels.csv"
traindf=pd.read_csv(csv_file)
testdf=pd.read_csv("newDat/true_labels.csv")
genres = ["action", "adventure", "animation", "comedy", "crime", "drama", "fantasy", "horror", "mystery", "romance", "sci-fi", "short"]

datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,validation_split=0.25)
train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./newDat/images",
    x_col="Image",
    y_col=genres,
    class_mode="raw",
    subset="training",
    batch_size=64,
    seed=1109,
    shuffle=True,
    target_size=(100,100))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./newDat/images",
    x_col="Image",
    y_col=genres,
    class_mode="raw",
    batch_size=32,
    seed=42,
    shuffle=True,
    target_size=(100,100))
"""
test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="newDat/test/images",
    x_col="id",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(32,32))
"""

data_aug = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])


model = models.Sequential([
    data_aug,

    layers.experimental.preprocessing.Rescaling(1./255), #does what the flatten method would do
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100,100,3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(512, activation='tanh'),
    layers.Dense(12, activation='sigmoid')
])

"""if you want to use "categorical_crossentropy", the labels should be one-hot-encoded. When your labels are given 
as an integer, changing to "sparse_categorical_crossentropy" is required. The advantage of using 
"categorical_crossentropy" is that it can give you class probabilities, which might be useful in some cases."""
op = tf.keras.optimizers.Adam(
    learning_rate=0.0001, #tried other vals and this seems best
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None)
model.compile( optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6),
              loss = ["binary_crossentropy","binary_crossentropy","binary_crossentropy",
                      "binary_crossentropy","binary_crossentropy","binary_crossentropy",
                      "binary_crossentropy","binary_crossentropy","binary_crossentropy",
                      "binary_crossentropy","binary_crossentropy","binary_crossentropy"],
              metrics=['accuracy'])

model.fit(x=train_generator,validation_data=valid_generator, epochs=25)

#do the actual training
#model.fit(train_x, train_y, epochs=25)

modelname = "movieclassifyer"
model.save(modelname, include_optimizer=False)