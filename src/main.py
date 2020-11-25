import numpy as np
import pickle

import os
import pyedflib
import tensorflow as tf
import utils.get_data as dt
import utils.models as models
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

OUTPUT_NUM = 6


def training():
    dataset = dt.Dataset(root_path='')

    sleep_model = models.get_resnet_model(output_number=OUTPUT_NUM)
    sleep_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    for partition in range(32):
        X_train, y_train = dataset.get_eeg_data(partition, data_type='train')
        X_val, y_val = dataset.get_eeg_data(np.random.randint(16), data_type='validation')
        X_train, y_train = shuffle(X_train, to_categorical(y_train, OUTPUT_NUM))
        y_val = to_categorical(y_val, OUTPUT_NUM)

        print("Train shape: {}, Validation shape: {}".format(X_train.shape, X_val.shape))
        sleep_model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_val, y_val))

        sleep_model.save('outputs/model.h5')

    sleep_model.save('outputs/model.h5')


if __name__ == '__main__':
    training()
