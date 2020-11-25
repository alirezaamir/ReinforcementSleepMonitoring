import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, splitext
import xml.etree.ElementTree as ET
import scipy.stats as ST

LEN = 2000
PART_LEN = 40
SR = 125
SEC_CHUNK = LEN//SR


class Dataset:
    filenames_dict = {'train': None, 'validation': None, 'test': None}

    def __init__(self, root_path):
        self.root_path = root_path

    def get_edf_files(self, data_type='train'):
        path = self.root_path + 'input/shhs2/' + data_type
        filenames = [f for f in listdir(path) if isfile(join(path, f))]
        return filenames

    def get_annotation(self, filename):
        base_name = splitext(filename)[0]
        path = self.root_path + 'input/annotation/' + base_name + '-nsrr.xml'
        tree = ET.parse(path)
        root = tree.getroot()
        # Get file duration
        duration = 0
        for child in root.iter('ScoredEvent'):
            if child.find('EventConcept').text == "Recording Start Time":
                duration = child.find('Duration').text
                duration = float(duration)
                duration = int(duration)

        labels = np.zeros(duration)
        #
        for child in root.iter('ScoredEvent'):
            if child.find('EventType').text == "Stages|Stages":
                start = float(child.find('Start').text)
                start = int(start)
                stage_duration = float(child.find('Duration').text)
                stage_duration = int(stage_duration)
                stage = child.find('EventConcept').text
                stage = int(stage[-1])
                labels[start:start+stage_duration] = stage
                # print("Start:{},\nDuration:{}\nstage:{}\n".format(start, stage_duration, stage))
        return labels

    def get_eeg_data(self, part_num, data_type=' train'):
        if self.filenames_dict[data_type] is None:
            self.filenames_dict[data_type] = self.get_edf_files(data_type)

        path = self.root_path + 'input/shhs2/' + data_type +'/'
        data = self.filenames_dict[data_type]
        X = np.zeros((0, LEN * 2))
        y = np.zeros((0))
        for patient_filename in data[PART_LEN * part_num:PART_LEN * (part_num + 1)]:
            print("PATIENT: {}".format(patient_filename))
            signals, signal_headers, header = pyedflib.highlevel.read_edf(path + patient_filename)
            extra_len = len(signals[2]) % LEN
            eeg1 = signals[2]
            eeg2 = signals[7]
            eeg1 = np.reshape(eeg1[:-extra_len], (-1, LEN))
            eeg2 = np.reshape(eeg2[:-extra_len], (-1, LEN))
            eeg_concatenated = np.concatenate((eeg1, eeg2), axis=1)

            total_label = self.get_annotation(patient_filename)

            label_extra_len = len(total_label) % SEC_CHUNK
            total_label = total_label[:-label_extra_len]
            total_label = np.reshape(total_label, (-1, SEC_CHUNK))
            labels = ST.mode(total_label, axis=1)[0]
            labels = np.squeeze(labels)
            labels = np.clip(labels, 0, 5)

            print("X shape: {}, y shape: {}".format(eeg_concatenated.shape, labels.shape))
            if eeg_concatenated.shape[0] == labels.shape[0]:
                X = np.concatenate((X, eeg_concatenated), axis=0)
                y = np.concatenate((y, labels), axis=0)
            else:
                print("Shape inconsistent in {}".format(patient_filename))

        return X, y


if __name__ == '__main__':
    dataset = Dataset(root_path='../')
    X, y = dataset.get_eeg_data(0, data_type='train')
    print("X: {}, y:{}".format(X.shape, y.shape))
