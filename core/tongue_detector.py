import os
from time import time
import json
import base64
import numpy as np

import torch
import cv2

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(BASE_PATH)), 'yolov5_tongue')
yolo_dir = os.path.join(BASE_PATH, 'yolov5')
best_path = os.path.join(yolo_dir, 'runs', 'train', 'exp15', 'weights', 'best.pt')

class TongueDetector:
    def __init__(self, image = None, yolo_dir = None, weights_path = None, ):
        if image is None:
            self.image = os.path.join(DATA_PATH, 'data', 'images', 'test', 'PICT0015.JPG')
        else:
            self.image, self.width, self.height = self.readb64(image)
        print('type(self.image): ', type(self.image))
        print('len(self.image): ', len(self.image))
        print('self.image: ', self.image)
        self.yolo_dir = yolo_dir if yolo_dir is not None else os.path.join(BASE_PATH, 'yolov5')
        self.weights_path = weights_path if weights_path is not None else os.path.join(self.yolo_dir, 'runs', 'train', 'exp15', 'weights', 'best.pt')
        self.model = torch.hub.load(self.yolo_dir, 'custom', self.weights_path, source='local')  # or yolov5n - yolov5x6, custom

    def readb64(self, image_b64_bytes: bytes):
        """
        Convert b64-bytes image to a numpy array,
        and return array dimensions
        """

        nparr = np.fromstring(base64.b64decode(image_b64_bytes), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w, c = img.shape
        return img, w, h

    def infer(self, image = None, show = True):
        start = time()
        if image is None:
            image = self.image
        results = self.model(image)
        if show:
            results.show()
        end = time()
        print(f'Time elapsed: {round(end-start, 2)}s')
        return results

if __name__ == '__main__':
    td = TongueDetector()
    td.infer()