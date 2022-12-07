"""Flask app for performing OMR and OCR on the questionnaire"""

from flask import Flask, request, render_template, jsonify, has_request_context
from flask.logging import default_handler
import logging
from logging.config import dictConfig

from core.tongue_detector import TongueDetector

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
app.config['JSON_SORT_KEYS'] = False # Necessary as some prediction keys are string and some are ints, so ordering becomes complex


@app.route('/')
def home():
    return render_template('test_deeplink.html')


@app.route('/detection', methods=['POST'])
def markers():
    omr_dict = {}
    ocr_dict = {}
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
    # CODE HERE
    print('type(b64_bytes): ', type(b64_bytes))
    td = TongueDetector(b64_bytes)
    results = td.infer()

    # CODE HERE
    ################
    response_dict = {}
    response_dict['success'] = True
    print(response_dict)
    return jsonify(response_dict)


if __name__ == '__main__':
    app.run()