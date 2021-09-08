import os
from flask import (
    Flask, 
    redirect, 
    render_template,
    request,
    jsonify,
)
from werkzeug.utils import secure_filename
from ClickCrop.yolo import (
    get_bounding_boxes,
)
from ClickCrop.polyrnn import select_bbox, calc_polygon
from entities import Polygon, db, Image as DbImage, BoundingBox
from PIL import Image
import uuid
import pathlib
# import models
import json
import numpy as np

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "./imgs"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
db.init_app(app)
with app.app_context():
    # db.drop_all()
    db.create_all()


@app.route('/')
def index():
    return "Hello world"

@app.route('/images', methods=['POST', 'GET'])
def upload_image():
    if request.files:
        image = request.files["image"]
        img_name = uuid.uuid4().hex + pathlib.Path(image.filename).suffix
        img_path = os.path.join(app.config["IMAGE_UPLOADS"], img_name)
        image.save(img_path)
        img_record = DbImage(
            name=image.filename,
            path=img_path,
        )
        db.session.add(img_record)
        db.session.commit()
        return {
            'id': img_record.id, 
            'name': img_record.name,
            'path': img_path,
        }, 201
        # return redirect(request.url)
    return {'error': 'Empty image'}, 400
    # return render_template("upload_image.html")

@app.route('/images/<img_id>/bounding-boxes', methods=['GET'])
def get_bbox(img_id):
    image = DbImage.query.get(img_id)
    objects = get_bounding_boxes(image.path)
    for obj in objects:
        bbox = BoundingBox(
            obj_class=obj["class"],
            index=obj["index"],
            confidence=obj["confidence"],
            width=obj["width"],
            height=obj["height"],
            center_x=obj["center"][0],
            center_y=obj["center"][1],
            bbox_x1=obj["box"][0][0],
            bbox_y1=obj["box"][0][1],
            bbox_x2=obj["box"][1][0],
            bbox_y2=obj["box"][1][1],
            image_id=image.id
        )
        db.session.add(bbox)
        db.session.commit()
    return {"data": objects}, 200

@app.route('/images/<img_id>/polygons', methods=['GET'])
def get_polygon(img_id):
    image = DbImage.query.get(img_id)
    x_coord = int(request.args.get('x'))
    y_coord = int(request.args.get('y'))
    roi = select_bbox(image, x_coord, y_coord)
    polygon = calc_polygon(roi)
    # print("POLYGONNNN:", polygon)
    # print("TYPEEE:", type(polygon))
    print("POLYGONNNN:", polygon)
    print("TYPEEE:", type(polygon))
    # polygon = polygon.tolist()
    # polygon = json.dumps({"coordinates": polygon})
    # polygon = np.array(polygon).tolist()
    poly = Polygon(
        poly_vertices=json.dumps(polygon[0].tolist()),
        image_id=image.id
    )
    db.session.add(poly)
    db.session.commit()
    return {"data": 
        {
            'id': poly.id,
            'coordinates': polygon[0].tolist(),
            'image': poly.image_id,
        }
    }, 200

if __name__ == '__main__':
    app.run(debug=True)
