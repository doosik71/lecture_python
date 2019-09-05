"""
Fit network model.
"""

import cv2
import json
import common
import numpy as np
import os
import os.path
import random


class ImageReader:
    """Read multiple images and convert to vector."""

    def __init__(self, image_width, image_height):
        """Set width and height."""

        self.width = image_width
        self.height = image_height

    def read_from(self, data_dir):
        """Read image data from given path."""

        # 파일 목록을 얻는다.
        file_list = [os.path.join(data_dir, f) for f
                    in os.listdir(data_dir)
                    if os.path.isfile(os.path.join(data_dir, f))]

        # 영상 벡터 목록으로 변환한다.
        return [common.read_image(f, self.width, self.height)
                for f in file_list]


class TrainData:
    """Convert positive and negative samples into training set."""

    def __init__(self, positive_samples, negative_samples):
        """Set positive samples and negative samples."""

        # 영상.
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

        # 영상의 개수.
        self.positive_count = len(positive_samples)
        self.negative_count = len(negative_samples)

        # 입력과 출력.
        self.x = self.positive_samples + self.negative_samples
        self.y = list(np.ones((self.positive_count,), np.float32)) + \
                 list(np.zeros((self.negative_count,), np.float32))

        # 입력과 출력을 합친다.
        self.data = list(zip(self.x, self.y))

        # 순서를 뒤섞는다.
        random.shuffle(self.data)

        # 입력값을 얻는다.
        self.x = np.array([item[0] for item in self.data])

        assert not np.isnan(self.x).any(), "Invalid input!"

        # 출력값을 얻는다.
        self.y = np.array([item[1] for item in self.data])

        assert not np.isnan(self.y).any(), "Invalid output!"


class Fitter():
    """Fit network model."""

    def __init__(self, config_path):
        # 설정 값을 읽는다.
        print('Reading config file...')
        self.config = common.Config(config_path)

        # 학습 영상 파일들을 읽는다.
        print('Reading image files...')
        image_reader = ImageReader(self.config.image_width, self.config.image_height)

        positive_samples = image_reader.read_from(self.config.positive_data_path)
        negative_samples = image_reader.read_from(self.config.negative_data_path)

        # 학습 데이터 형태로 변환한다.
        print('Preparing training files...')
        self.train_data = TrainData(positive_samples, negative_samples)

        print('# of images = {} positives vs. {} negatives'.format(
            self.train_data.positive_count,
            self.train_data.negative_count))

        # 학습 모델을 생성한다.
        print('Building training model...')
        shape = (self.config.image_height, self.config.image_width, 3)
        self.model = common.get_model(shape, self.config.model_type)

    def fit(self):
        # 모델을 학습한다.
        print('Training model...')
        self.model.fit(x=self.train_data.x, y=self.train_data.y, epochs=self.config.epochs)

        # 학습된 모델을 파일에 저장한다.
        print('Writing model...')
        self.model.save_weights(self.config.model_path)
        self.model.summary()


if __name__ == "__main__":

    fitter = Fitter('config.json')
    fitter.fit()
