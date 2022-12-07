import os
import unittest
import requests
import base64

import logging

LOCAL=0
PROD=1
DEV=2

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestTongueDetectProd(unittest.TestCase):
    def __test_tongue(self, case, noImageError=False):
        filename = os.path.join(BASE_PATH, 'niigata_practise', '1_tongue', 'PICT0133.JPG')
        with open(filename, "rb") as image_file:
            # print(f"{name[1]}_3.jpg")
            encoded_bytes = base64.b64encode(image_file.read())
            header = 'data:image/jpg;base64,'
            base64_string = header+encoded_bytes.decode('utf-8')
    
            if case==LOCAL:
                urlBase = "http://localhost:5000/"
            elif case==PROD:
                urlBase = "https://___.azurewebsites.net/"
            elif case==DEV:
                urlBase = "https://___.azurewebsites.net/"

            path = 'detection'
            
            if noImageError:
                myobj = {}
            else:
                myobj = {
                    'image': base64_string
                    }

            response = requests.post(urlBase+path, json = myobj)
            response.raise_for_status()  # raises exception when not a 2xx response
            if response.status_code != 204:
                print('jsoning response')
                res = response.json()

            # Check the output.
            if noImageError:
                self.assertTrue(
                    'error' in res,
                )
            else:
                self.assertTrue(
                    'success' in res,
                )


    def test_tongue(self):
        self.__test_tongue(PROD)


    def test_tongue_noImageError(self):
        self._TestTongueDetectProd__test_tongue(PROD, True)

    
class TestTongueDetectLocal(TestTongueDetectProd):
    def test_tongue(self):
        self._TestTongueDetectProd__test_tongue(LOCAL)


    def test_tongue_noImageError(self):
        self._TestTongueDetectProd__test_tongue(LOCAL, True)


class TestTongueDetectDev(TestTongueDetectProd):
    def test_tongue(self):
        self._TestTongueDetectProd__test_tongue(DEV)


    def test_tongue_noImageError(self):
        self._TestTongueDetectProd__test_tongue(DEV, True)