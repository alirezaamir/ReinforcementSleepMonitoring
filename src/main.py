import numpy as np
import pickle

import os
import pyedflib
import tensorflow as tf
import utils.get_data as dt
import utils.models as models
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from tensorflow.keras.utils import to_categorical
import datetime

OUTPUT_NUM = 5
version_code = 5


def training():
    dataset = dt.Dataset(root_path='')
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    sleep_model = models.get_resnet_model(output_number=OUTPUT_NUM)
    sleep_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    for partition in range(160):
        X_train, y_train = dataset.get_eeg_data(partition % 32, data_type='train')
        X_train = scale(X_train, axis =1)
        print("std:{}".format(np.std(X_train[1,:])))
        X_val, y_val = dataset.get_eeg_data(np.random.randint(16), data_type='validation')
        X_val = scale(X_val, axis=1)
        X_train, y_train = shuffle(X_train, to_categorical(y_train, OUTPUT_NUM))
        y_val = to_categorical(y_val, OUTPUT_NUM)

        print("Train shape: {}, Validation shape: {}".format(X_train.shape, X_val.shape))
        sleep_model.fit(X_train, y_train, batch_size=64, epochs=10*(partition+1), initial_epoch=10*partition,
                        validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

        sleep_model.save('outputs/model_v{}'.format(version_code))

    sleep_model.save('outputs/model_v{}'.format(version_code))


if __name__ == '__main__':
    training()
