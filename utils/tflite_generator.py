import tensorflow as tf
from tensorflow.keras.models import load_model


def save():
    root_dir = '../outputs/'
    saved_model = root_dir + 'model_v3'

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model)
    tflite_model = converter.convert()
    open(root_dir + 'model.tflite', 'wb').write(tflite_model)


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    save()