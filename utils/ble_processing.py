import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt


def butter_highpass_filter(data, low, high, fs, order= 3):
    nyq = 0.5 * fs  # Nyquist Frequency
    low_cutoff = low / nyq
    high_cutoff = high / nyq
    # Get the filter coefficients
    b, a = butter(order, [low_cutoff, high_cutoff], btype='bandpass', analog=False)
    y = filtfilt(b, a, data)
    return y


def read_data():
    file_name = '../input/ble/eeg_v1.txt'
    with open(file_name, 'rb') as fp:
        line_list = []
        lines = fp.readlines()
        for l in lines:
            line_list.append(l.hex())
        raw_data = "".join(line_list)
        index_list= []
        index_pre = 0
        data = []
        for i in range(1404):  # 1404
            pack2000 = raw_data[4002 * i + 2:4002 * (i + 1)]
            for j in range(100):
                index = int(pack2000[40*j:40*(j+1)][2:4], 16)
                if index == 0 and index_pre != 255:
                    # print("Error in {}, {}: {}, {} : {}".format(i, j, index, index_pre, pack2000[40*j:40*(j+1)]))
                    index_pre = index
                    continue
                for d in range(16):
                     data.append(int(pack2000[40*j:40*(j+1)][2*d+4:2*(d+1)+4], 16))
                # print(pack2000[40*j:40*(j+1)])
                index_list.append(index)

                index_pre = index

        data = np.array(data)
        print(data.shape)
        # flatten_data = np.reshape(data, (-1,))[16:]
        two_channels_data = np.reshape(data[:-32], (-1,6))
        EEG1 = two_channels_data[:,:3]
        EEG2 = two_channels_data[:,3:]
        EEG1_samples = EEG1[:,2] + 256*( EEG1[:,1] + 256*EEG1[:,0])
        EEG2_samples = EEG2[:,2] + 256*( EEG2[:,1] + 256*EEG2[:,0])
        EEG1_twos_comp = np.where(EEG1[:, 0] > 127, EEG1_samples - 256*256*256, EEG1_samples)
        EEG2_twos_comp = np.where(EEG2[:, 0] > 127, EEG2_samples - 256*256*256, EEG2_samples)
        return EEG1_twos_comp, EEG2_twos_comp


def filtering_and_processing(signal):
    EEG_remove_spikes = signal[np.abs(signal) < 2e6]

    zero_crossing_points = [i for i in range(1, len(EEG_remove_spikes)) if
                            EEG_remove_spikes[i - 1] * EEG_remove_spikes[i] < 0]
    zero_crossing_points.insert(0, 0)

    EEG_filtered = np.zeros(0)
    for point_idx in range(1, len(zero_crossing_points)):
        start = zero_crossing_points[point_idx - 1] + 2
        end = zero_crossing_points[point_idx] - 2
        chunk = EEG_remove_spikes[start:end]
        filtered = butter_highpass_filter(chunk, 1, 30, 125)
        EEG_filtered = np.concatenate((EEG_filtered, filtered))

    return EEG_filtered


def main():
    EEG1 , EEG2 = read_data()

    EEG1_filtered = filtering_and_processing(EEG1)
    EEG2_filtered = filtering_and_processing(EEG2)

    print(EEG1_filtered.shape)



    plt.subplot(211)
    plt.plot(EEG1[:2000])
    plt.subplot(212)
    plt.plot(EEG1_filtered[:2000])

    plt.figure()
    plt.subplot(211)
    plt.plot(EEG2[:2000])
    plt.subplot(212)
    plt.plot(EEG2_filtered[:2000])
    plt.show()

    # print(EEG2[150000:150200])


if __name__ == '__main__':
    main()

