"""Flask app for performing detection, slicing,
and categorisation on patient tongue images"""

import base64
import logging
from logging.config import dictConfig
import os
from time import time

from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from flask import Flask, request, render_template, jsonify, has_request_context
from flask.logging import default_handler

from core.classifier import TongueClassifier
from core.detector import TongueDetector
from core.helpers import reshape_split, image_tile

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


class RequestFormatter(logging.Formatter):
    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
        else:
            record.url = None
            record.remote_addr = None
        return super().format(record)


formatter = RequestFormatter(
    '[%(asctime)s] %(remote_addr)s requested %(url)s\n'
    '%(levelname)s in %(module)s: %(message)s'
)
default_handler.setFormatter(formatter)


app = Flask(__name__)
# app.json_provider_class = LazyJSONEncoder

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'niigataAPI',
            "route": '/niigataAPI.yaml',
            "version": "0.0.1",
            "title": "NiigataAPI",
            "description": 'API for TCI classification',
            # "rule_filter": lambda rule: True,
            # "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    # "swagger_ui": True,
    # "specs_route": "/apidocs/"
}


swagger = Swagger(
    app,
    # template=swagger_template,             
    config=swagger_config
    )
@swag_from(os.path.join("core", "yaml", "home.yaml"))
@app.route('/')
def home():
    return render_template('test_deeplink.html')

@swag_from(os.path.join("core", "yaml", "inference.yaml"), methods=['POST'])
@app.route('/inference', methods=['POST'])
def markers():
    start = time()
    logging.debug('start POST /markers')
    json_data = request.get_json()
    # Guard
    if 'image' not in json_data:
        logging.warning('Request does not have "image" argument.')
        response_dict = {'success': False, 'error': 'Request does not have "image" argument.'}
        return jsonify(response_dict)
    logging.debug('request has image args.')
    # Remove b64string headers
    b64_string = json_data['image'].split('base64,')[1]
    b64_bytes = b64_string.encode('utf-8')
    print('Passing image for object detection')

    ################
    # ML CODE HERE
    t1 = time()
    global td, tc
    image_out, results = td.run(b64_bytes)
    if image_out is None:
        logging.warning('Unable to identify tongue in image.')
        response_dict = {'success': False, 'error': 'Unable to identify tongue in image.'}
        return jsonify(response_dict)
    tiled_image = reshape_split(image_out)
    tcis = tc.run(tiled_image)
    t2 = time()
    out_im = image_tile(image_out, tcis)
    filename = os.path.join('outputs', 'tiled.png')
    out_im.savefig(filename)
    print(type(out_im))
    # ML CODE HERE
    ################

    with open(filename, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        header = 'data:image/png;base64,'
        base64_string = header+encoded_bytes.decode('utf-8')

    inference_time = round(t2-t1, 2)
    response_dict = {}
    response_dict['categories'] = tcis
    response_dict['image_segments'] = base64_string
    response_dict['success'] = True
    end = time()
    total_time = round(end-start, 2)
    print('Response dict keys:', response_dict.keys())
    print(f'Model inference time: {inference_time}s')
    print(f'Total elapsed time: {total_time}s')
    return jsonify(response_dict)

if __name__ == '__main__':
    td = TongueDetector()
    tc = TongueClassifier()
    app.run()