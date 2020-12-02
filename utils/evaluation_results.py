import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import utils.get_data as dt


from tensorflow.keras.models import load_model


def main():
    sleep_model = load_model('../outputs/model.h5')
    dataset = dt.Dataset('../')
    predict_list = np.zeros((0, 6))
    groundtruth = np.zeros(0)
    for partition in range(16):
        X_test, y_test = dataset.get_eeg_data(partition, 'test')
        predict = sleep_model.predict(X_test)
        predict_list = np.concatenate((predict_list, predict))
        groundtruth = np.concatenate((groundtruth, y_test))

    print("Predict shape: {}, groundtruth: {}".format(predict_list.shape, groundtruth.shape))
    print("Conf Mat: {}".format(confusion_matrix(groundtruth, np.argmax(predict_list,axis=1))))


if __name__ == '__main__':
    main()