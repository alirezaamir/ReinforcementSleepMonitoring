import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import utils.get_data as dt
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import butter,filtfilt
import pickle


def butter_lowpass_filter(data, cutoff, fs, order= 3):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def get_confusion_matrix():
    sleep_model = load_model('../outputs/model_v4')
    dataset = dt.Dataset('../')
    predict_list = np.zeros((0, 5))
    groundtruth = np.zeros(0)
    for partition in range(16):
        X_test, y_test = dataset.get_eeg_data(partition, 'test')
        predict = sleep_model.predict(X_test)
        predict_list = np.concatenate((predict_list, predict))
        groundtruth = np.concatenate((groundtruth, y_test))

    print("Predict shape: {}, groundtruth: {}".format(predict_list.shape, groundtruth.shape))
    print("Conf Mat: {}".format(confusion_matrix(groundtruth, np.argmax(predict_list,axis=1))))


def visualize_output():
    sleep_model = load_model('../outputs/model_v4')
    dataset = dt.Dataset('../')
    X_test, y_test = dataset.get_eeg_data(part_num=1, data_type='test', single_patient=True)
    predict = sleep_model.predict(X_test[:])
    predict_class = np.argmax(predict, axis=1)
    print(predict_class.shape)
    # print("std:{}, mean:{}".format(np.std(X_test[:50], axis=1), np.mean(X_test[:50], axis=1)))
    fs = 1/16  # sample rate, Hz
    cutoff = 1/2400
    filtered_predict = butter_lowpass_filter(predict_class, cutoff, fs)

    plt.figure()
    plt.subplot(311)
    plt.plot(y_test[:])
    plt.subplot(312)
    plt.plot(predict_class)
    plt.subplot(313)
    plt.plot(filtered_predict)
    plt.show()


def predict_ble():
    sleep_model = load_model('../outputs/model_v4')
    filename = '../input/ble/eeg.pickle'
    data = pickle.load(open(filename, 'rb'))
    eeg_data = data['eeg']
    print("shape: {}".format(eeg_data.shape))

    predict = sleep_model.predict(eeg_data)
    predict_class = np.argmax(predict, axis=1)
    print("predict shape:{}".format(predict.shape))
    pickle.dump({'predict': predict}, open('../outputs/predict.pickle', 'wb'))
    plt.plot(predict_class)
    plt.show()


if __name__ == '__main__':
    # get_confusion_matrix()
    # tf.config.experimental.set_visible_devices([], 'GPU')
    # visualize_output()
    predict_ble()
