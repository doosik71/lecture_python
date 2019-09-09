"""Network model."""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # pylint: disable=import-error
from tensorflow.keras.layers import Dense, Flatten  # pylint: disable=import-error
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # pylint: disable=import-error


class Config:
    """Configuration information."""

    def __init__(self, config_path):
        """Read config file."""

        with open(config_path) as f:
            # 설정 파일을 읽는다.
            data = json.load(f)

            # 읽은 값을 멤버 변수로 저장한다.
            self.model_type = data['model_type']
            self.positive_data_path = data['positive_data_path']
            self.negative_data_path = data['negative_data_path']
            self.image_width = int(data['image_width'])
            self.image_height = int(data['image_height'])
            self.epochs = int(data['epochs'])
            self.model_path = data['model_path']


def get_model(shape, model_type=None):
    """Get model."""

    if model_type == 'mlp':
        # 모델을 생성한다.
        model = Sequential()
        model.add(Flatten(input_shape=shape))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
    elif model_type == 'cnn':
        # 모델을 생성한다.
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=10, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
    else:
        assert False, 'Invalid model type!'

    # 생성된 모델을 반환한다.
    return model


def read_image(image_path, width, height):
    """Read a image."""

    # 영상을 읽는다.
    image = plt.imread(image_path)  # pylint: disable=no-member

    # 영상을 읽지 못하면, 오류를 출력한다.
    assert image is not None, "Cannot read " + image_path

    if len(image.shape) == 2:  # 흑백 영상이면,
        # 컬러 영상으로 변환한다.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # pylint: disable=no-member
    elif image.shape[2] == 4:  # RGBA 영상이면,
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # pylint: disable=no-member

    # 원본 영상의 크기를 얻는다.
    h, w, _ = image.shape

    # 영상의 크기가 원하는 크기가 아니면,
    if h != height or w != width:
        # 영상의 크기를 원하는 크기로 변경한다.
        image = cv2.resize(image, (height, width))  # pylint: disable=no-member

    # 영상의 픽셀 값을 0~1 사이의 값으로 변환하고, 변환된 영상 벡터를 반환한다.
    return image.astype(np.float32) / 255.0
