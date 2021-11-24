import numpy as np
import cv2
from random import randint
import utils.constants as constants
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
import time
import pandas as pd

# GENERIC UTILS

def load_quickdrow_images_from_npy(npy_file_directory :str) -> list[list[int]]:
    images = np.load(npy_file_directory)
    return images

def plot_quickdraw_image(image_array: list[int]) -> None:
    image_inverted_reshaped = cv2.bitwise_not(image_array).reshape(28,28)
    plt.imshow(image_inverted_reshaped, cmap='gray')
    plt.show()

def plot_random_quickdraw_image(images_array: list[list[int]]) -> None:
    image_random_position = randint(0, len(images_array))
    plot_quickdraw_image(images_array[image_random_position])

# PROCESS UTILS

def generate_x_y_from_dataset(data, classification, data_matrix, x, y):
    data_sample = data[:constants.NUMERO_DE_AMOSTRAS]
    for sample in data_sample:
        sample_processed = sample.astype(np.float32)
        sample_processed /= 255.0
        x.append(sample_processed.reshape((28, 28, 1)))
        y.append(data_matrix[classification])

# KERAS MODELS

def create_cnn_model(optimizer = 'adam'):
    modelo = Sequential()
    modelo.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28,28,1)))

    modelo.add(Conv2D(64, (3, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Dropout(0.5))                 
    modelo.add(Flatten())
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(constants.NUMERO_DE_CLASSES, activation='softmax'))
    modelo.compile(optimizer = optimizer, loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

    return modelo

def generate_excel_metrics_crossvalidate(x, y, build_model, file_name, n_splits = 10):
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    fit_time_values = []
    score_time_values = []
    kf = KFold(n_splits=n_splits, shuffle=True)

    y_single = np.argmax(y, axis=1)
    for i in range(0, 30):
        print(f"===============================================Actual cross validate: {i+1}==================================================================")
        actual_acc = []
        actual_precision = []
        actual_recall = []
        actual_f1 = []
        actual_fit_time = []
        actual_predict_time = []
        for train_index, val_index in kf.split(x):
            model = KerasClassifier(build_fn=build_model, epochs=50, verbose=0)
            fit_time = time.time()
            model.fit(x[train_index], y[train_index])
            end_fit_time = time.time()

            pred_time = time.time()
            pred = model.predict(x[val_index])
            end_pred_time = time.time()

            actual_acc.append(accuracy_score(y_single[val_index], pred))
            actual_precision.append(precision_score(y_single[val_index], pred, average='weighted'))
            actual_recall.append(recall_score(y_single[val_index], pred, average='weighted'))
            actual_f1.append(f1_score(y_single[val_index], pred, average='weighted'))
            actual_fit_time.append(end_fit_time - fit_time)
            actual_predict_time.append(end_pred_time - pred_time)

        accuracy_values.append(np.mean(actual_acc))
        precision_values.append(np.mean(actual_precision))
        recall_values.append(np.mean(actual_recall))
        f1_values.append(np.mean(actual_f1))
        fit_time_values.append(np.mean(actual_fit_time))
        score_time_values.append(np.mean(actual_predict_time))

    model_data = pd.DataFrame(data={"Acurácia": accuracy_values,
                               "Precisão": precision_values,
                               "Recall": recall_values,
                               "F1": f1_values,
                               "Tempo de treinamento": fit_time_values,
                               "Tempo de classificação": score_time_values})

    model_data.to_excel(f"metrics/{file_name}")