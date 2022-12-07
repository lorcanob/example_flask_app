"""The TongueDetector class takes in an image including a tongue and
returns a cropped, resized bounding image to be tiled and passed
to the TongueClassifier
"""

import os
import base64
import sys
from time import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

from helpers import BASE_PATH


class TongueDetector:
    """YOLO based object detector that detects, crops, and resizes an input image
    centred on the tongue found in the image
    """
    def __init__(self, yolo_dir = None, weights_path = None):
        if yolo_dir is not None:
            self.yolo_dir = yolo_dir
        else:
            self.yolo_dir = os.path.join(BASE_PATH, 'yolov5')
        if weights_path is not None:
            self.weights_path = weights_path
        else:
            self.weights_path = os.path.join(
                self.yolo_dir, 'retrain', 'train', 'exp2', 'weights', 'best.pt'
                )
        self.model = torch.hub.load(
            self.yolo_dir, 'custom', self.weights_path, source='local'
            )  # or yolov5n - yolov5x6, custom

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        """Resize an image to either a height or width value without
        changine its aspect ratio
        """
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation = inter)
        return resized
    
    def readb64(self, image_b64_bytes: bytes, resize = False):
        """
        Convert b64-bytes image to a numpy array,
        and return array dimensions
        """
        nparr = np.frombuffer(base64.b64decode(image_b64_bytes), np.uint8)
        self.raw_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_RGB2BGR)
        self.image = self.raw_image.copy()
        self.height, self.width, c = self.image.shape
        if resize:
            self.image = self.image_resize(self.image, width=640)
            print("self.image.shape", self.image.shape)
        return self.image

    def infer(self, image = None, show = True):
        """Perform object detection on the input image
        and return the bounding box results
        """
        start = time()
        if image is None:
            image = self.image
        self.results = self.model(image)
        if show:
            self.results.show()
        end = time()
        print(f'Detector inference time: {round(end-start, 2)}s')
        return self.results

    def crop_and_square(self, image = None, results = None):
        """Crop the input image around the detected tongue and resize the
        image to a square to ensure numpy-friendly slicing when tiling
        """
        if image is None:
            image = self.image
        if results is None:
            results = self.results
        if results.pandas().xyxy[0].empty:
            return None
        xmin = round(results.pandas().xyxy[0]['xmin'][0])
        ymin = round(results.pandas().xyxy[0]['ymin'][0])
        xmax = round(results.pandas().xyxy[0]['xmax'][0])
        ymax = round(results.pandas().xyxy[0]['ymax'][0])
        image_out = image[ymin:ymax, xmin:xmax]
        dimensions = (7*224, 7*224)
        image_out = cv2.resize(image_out, dimensions)
        return image_out

    def run(self, image_b64_bytes: bytes, resize = False, show = False):
        """Read the input image, perform object detection inference,
        and crop the detected image
        """
        image = self.readb64(image_b64_bytes, resize)
        results = self.infer(image, show)
        image_out = self.crop_and_square(self.raw_image, results)
        return image_out, results

if __name__ == '__main__':
    td = TongueDetector()
    filename = os.path.join(BASE_PATH, os.pardir, os.pardir, 'niigata_practise', '1_tongue', 'PICT0133.JPG')
    with open(filename, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        base64_string = encoded_bytes.decode('utf-8')
    td.readb64(encoded_bytes)
    td.infer()