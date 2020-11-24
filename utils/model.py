
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D

vector_len = 4000


def my_residual_block(in_cnv, stage: str, filter_size, stride=1):
    bn1 = BatchNormalization(name='BN1_' + stage)(in_cnv)
    relu1 = ReLU(name='ReLU1_' + stage)(bn1)
    drop = Dropout(rate = 0.1, name='Drop_' + stage)(relu1)

    cnv2 = Conv1D(filter_size, strides=stride, kernel_size=5, padding='same', name='Conv2_' + stage)(drop)
    bn2 = BatchNormalization(name='BN2_' + stage)(cnv2)
    relu2 = ReLU(name='ReLU2_' + stage)(bn2)

    cnv3 = Conv1D(filter_size, strides=1, kernel_size=5, padding='same', name='Conv3_' + stage)(relu2)
    size_changed = (in_cnv.shape[2] != filter_size)
    if size_changed:
        shortcut = Conv1D(filter_size, kernel_size=1, strides=stride, name='Shortcut_'+stage, padding='same')(in_cnv)
        add = Add(name='Add_' + stage)([shortcut, cnv3])
    elif stride>1:
        pool = MaxPooling1D(pool_size=5, strides=stride, name='Pool_'+stage, padding='same')(in_cnv)
        add = Add(name='Add_' + stage)([pool, cnv3])
    else:
        add = Add(name='Add_' + stage)([in_cnv, cnv3])
    return add


def get_resnet_model(output_number=2):
    signal_input = Input(shape=(vector_len, 1))
    x1_cnv = Conv1D(32, kernel_size=5, strides=2, padding='same')(signal_input)
    res1 = my_residual_block(x1_cnv, '1', filter_size=64, stride=2)
    res2 = my_residual_block(res1, '2', filter_size=64, stride=2)
    res3 = my_residual_block(res2, '3', filter_size=128, stride=2)

    bn5 = BatchNormalization(name='BN_5')(res3)
    relu5 = ReLU(name='ReLU_5')(bn5)
    avg = AveragePooling1D(pool_size=5)(relu5)
    flatten = Flatten(name='Flat_5')(avg)
    drop1 = Dropout(rate=0.5)(flatten)
    dense1 = Dense(256, activation='relu', name='Dense1_5')(drop1)
    drop2 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(output_number, activation='softmax', name='Dense2_5')(drop2)

    model = Model(inputs=signal_input, outputs=dense2)
    return model


def plot(model: Model, name: str):
    plot_model(model, to_file='outputs/' + name + '.png', show_shapes=True, show_layer_names=False)


if __name__ == '__main__':
    model = get_resnet_model(2, True)
    plot(model, 'My_ResNet_like')
