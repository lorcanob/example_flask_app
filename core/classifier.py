"""The TongueClassifier class takes in a set of tongue image tiles and
returns a list of TCIs corresponding to each image"""

import os
from time import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import onnxruntime as rt

from helpers import BASE_PATH, reshape_split

resnet_model_path = os.path.join(BASE_PATH, 'weights', 'TongueCoating_Resnet18.onnx')
google_model_path = os.path.join(BASE_PATH, 'weights', 'TongueCoating_GoogLeNet.onnx')


class TongueClassifier:
    """Classifier that loops through a tiled array of subimages
    and returns a list of the categorised model predictions for
    each subimage
    """
    def __init__(self, model = 'resnet') -> None:
        if model == 'resnet':
            self.model_path = resnet_model_path
        elif model == 'google':
            self.model_path = google_model_path
        self.sess = rt.InferenceSession(self.model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.labels = ['0', '1', '2', '3', '4', '5']

    def infer(self, image = None):
        """For each subimage, perform inference and return the predicted category
        """
        X = np.asarray(image)
        X = X.transpose(2,0,1)
        X = X.reshape(1,3,224,224)
        out = self.sess.run(None, {self.input_name: X.astype(np.float32)})
        tci = out[0][0]
        return self.labels[tci.argmax(0)]

    def run(self, tiled_image):
        """Parse through each subimage of a tiled image and return a list
        of predicted categories
        """
        start = time()
        counter = 0
        tongue_cats = []
        
        for i in range(7):
            for j in range(7):
                counter+=1
                tongue_cat = self.infer(tiled_image[i, j])
                tongue_cats.append(tongue_cat)
        end = time()
        print(f"Classifier time elapsed: {round(end-start, 2)}s")
        return tongue_cats

if __name__ == '__main__':
    tc = TongueClassifier()
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(BASE_PATH)), 'niigata_practise')
    filename = os.path.join(BASE_PATH, os.pardir, os.pardir, 'niigata_practise', '1_tongue', 'PICT0133.JPG')
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, 4)
    image = cv2.resize(image, (7*224, 7*224))
    X = np.asarray(image)
    tiled_image = reshape_split(X)
    tc.run(tiled_image)
    