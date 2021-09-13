# import models
import json
import os
import pathlib
import uuid
import boto3

import numpy as np
from flask import Flask, request, render_template, redirect

from polyrnn import calc_polygon, draw_results, select_bbox
from yolo import get_bounding_boxes
from entities import BoundingBox
from entities import Image as DbImage
from entities import Polygon, db

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "imgs/uploads"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
s3_client = boto3.client('s3', config= boto3.session.Config(signature_version='s3v4'))
db.init_app(app)
with app.app_context():
    # db.drop_all()
    db.create_all()


@app.route('/')
def index():
    return {'message': 'Hola Mundo'}, 200

@app.route('/images', methods=['POST', 'GET'])
def upload_image():
    if request.files:
        image = request.files["image"]
        img_name = uuid.uuid4().hex + pathlib.Path(image.filename).suffix
        img_path = os.path.join(app.config["IMAGE_UPLOADS"], img_name)
        s3_client.put_object(Body=image, Bucket=os.getenv("BUCKET"), Key=img_path)
        # image.save(img_path)
        img_record = DbImage(
            name=image.filename,
            path=img_path,
        )
        db.session.add(img_record)
        db.session.commit()
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': os.getenv("BUCKET"),
                'Key': img_path
            },
            ExpiresIn=3600,
        )
        return {
            'id': img_record.id, 
            'name': img_record.name,
            'path': img_path,
            'url': url,
        }, 201
        # return redirect(request.url)
    # return {'error': 'Empty image'}, 400
    return render_template("upload_image.html")

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
    return {
        "data": objects,
            # "url": url,
    }, 200

@app.route('/images/<img_id>/polygons', methods=['GET'])
def get_polygon(img_id):
    image = DbImage.query.get(img_id)
    x_coord = int(request.args.get('x'))
    y_coord = int(request.args.get('y'))
    roi, coords = select_bbox(image, x_coord, y_coord)
    polygon, crop_path = calc_polygon(roi)
    s3_client.upload_file(crop_path, os.getenv("BUCKET"), crop_path)
    results_path = draw_results(image, polygon, coords)
    s3_client.upload_file(results_path, os.getenv("BUCKET"), results_path)
    poly = Polygon(
        poly_vertices=json.dumps(polygon.tolist()),
        image_id=image.id
    )
    db.session.add(poly)
    db.session.commit()
    crop_url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': os.getenv("BUCKET"),
            'Key': crop_path
        },
        ExpiresIn=3600,
    )
    results_url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': os.getenv("BUCKET"),
            'Key': results_path
        },
        ExpiresIn=3600,
    )
    return {
        "data": {
            'id': poly.id,
            'coordinates': polygon.tolist(),
            'image': poly.image_id,
            'crop_url': crop_url,
            'results_url': results_url,
        }
    }, 200

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=True if os.getenv("DEBUG") == "True" else False,
        port=os.getenv("PORT"),
    )
