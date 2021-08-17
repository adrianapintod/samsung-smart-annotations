import os

from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, url_for)
from werkzeug.utils import secure_filename

from ClickCrop.yolo import (
    get_colors, 
    get_config, 
    get_labels, 
    get_prediction,
    get_weights, 
    load_image, 
    load_model,
    get_bounding_boxes,
)

from PIL import Image

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "./imgs"

@app.route('/')
def index():
    return "Hello world"

@app.route('/home')
def print_name(name):
    return "Hi, {}".format(name)

@app.route('/images', methods=['POST', 'GET'])
def upload_image():
    if request.files:
        image = request.files["image"]
        print(image)
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
        # return {'name': image.filename}, 201
        return redirect(request.url)

    # return {'error': 'Empty image'}, 400
    return render_template("upload_image.html")

@app.route('/images/<img_id>/bounding-boxes', methods=['GET'])
def get_bbox(img_id):
    img = os.path.join(app.config["IMAGE_UPLOADS"], img_id)
    get_bounding_boxes(img)
    return Response(response={'hola': 'mundo'}, status=200)

if __name__ == '__main__':
    app.run(debug=True)
