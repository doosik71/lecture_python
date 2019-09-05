"""
Predict output by trained model.
"""

import cv2
import json
import common
import numpy as np
import os
import os.path
import random


class Predictor:
    def __init__(self, config_path):
        """Load trained_model."""

        # 환경 설정 값을 읽는다.
        print('Reading config file...')
        self.config = common.Config(config_path)

        # 학습 모델을 읽는다.
        print('Reading model...')
        shape = (self.config.image_height, self.config.image_width, 3)
        self.model = common.get_model(shape, self.config.model_type)
        self.model.load_weights(self.config.model_path)
        self.model.summary()

    def predict(self, image_file_path):
        """Predict output by trained model."""

        # 영상을 읽는다.
        print('Reading image...')
        x = common.read_image(image_file_path,
                              self.config.image_width,
                              self.config.image_height)

        # 모델을 이용하여 결과를 얻는다.
        return self.model.predict(np.array([x]))


if __name__ == "__main__":

    predictor = Predictor('config.json')

    print(predictor.predict('./data/positive/2.png'))
    print(predictor.predict('./data/negative/1.png'))
