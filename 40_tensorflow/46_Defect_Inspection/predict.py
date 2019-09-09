"""
Predict output by trained model.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import json
import common
import numpy as np
import os
import os.path
import random
import sys
import time


class Predictor:
    def __init__(self, config_path):
        """Load trained_model."""

        # 환경 설정 값을 읽는다.
        print('Reading config file...')
        self.config = common.Config(config_path)

        # 학습 모델을 읽는다.
        shape = (self.config.image_width, self.config.image_height, 3)
        self.model = common.get_model(shape, self.config.model_type)
        self.model.load_weights(self.config.model_path)
        # self.model.summary()

    def predict(self, image_file_path):
        """Predict output by trained model."""

        # 영상을 읽는다.
        x = common.read_image(image_file_path,
                              self.config.image_width,
                              self.config.image_height)

        # 모델을 이용하여 결과를 얻는다.
        return self.model.predict(np.array([x]))


if __name__ == "__main__":

    start = time.time()
    predictor = Predictor('config.json')
    loading_time = time.time() - start
    print('Model loading time =', loading_time)

    if len(sys.argv) >= 2:
        for image_path in sys.argv[1:]:
            print('Input image =', image_path)
            start = time.time()
            print('Prediction results =', predictor.predict(image_path))
            processing_time = time.time() - start
            print('Processing time =', processing_time)

