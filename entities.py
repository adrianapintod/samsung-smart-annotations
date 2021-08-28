from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    path = db.Column(db.String(500), nullable=False)

class BoundingBox(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    obj_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    index = db.Column(db.Integer, nullable=False)
    bbox_x1 = db.Column(db.Integer, nullable=False)
    bbox_y1 = db.Column(db.Integer, nullable=False)
    bbox_x2 = db.Column(db.Integer, nullable=False)
    bbox_y2 = db.Column(db.Integer, nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)

class Polygon(db.Model):
    id = db.Column(db.Integer, primary_key=True)
