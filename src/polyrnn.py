import os
import uuid
from math import hypot

import matplotlib

matplotlib.use('agg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tensorflow as tf
import boto3

import utils
from entities import BoundingBox
from models_init import ggnnModel, ggnnSess, model, polySess
from poly_utils import vis_polys

s3_client = boto3.client('s3', config= boto3.session.Config(signature_version='s3v4'))

IMAGE_UPLOADS = "imgs/crops"
POLYGON_UPLOADS = "imgs/crops_poly"
RESULT_UPLOADS = "imgs/results"
POLYRNN_METAGRAPH='./src/checkpoints/polyrnn/poly/polygonplusplus.ckpt.meta'
POLYRNN_CHECKPOINT='./src/checkpoints/polyrnn/poly/polygonplusplus.ckpt'
EVALNET_CHECKPOINT='./src/checkpoints/polyrnn/evalnet/evalnet.ckpt'
GGNN_METAGRAPH='./src/checkpoints/polyrnn/ggnn/ggnn.ckpt.meta'
GGNN_CHECKPOINT='./src/checkpoints/polyrnn/ggnn/ggnn.ckpt'
_FIRST_TOP_K = 6

def distance(p1, p2):
#"""Euclidean distance between two points."""
	# print(p2)
	xx1, yy1 = p1
	xx2, yy2 = p2
	return hypot(xx2 - xx1, yy2 - yy1)

def select_bbox(img, x, y):
	s3_client.download_file(os.getenv("BUCKET"), img.path, img.path)
	image = cv2.imread(img.path)[:, :, ::-1]
	bounding_boxes = BoundingBox.query.filter_by(image_id=img.id).all()
	click_pos = (x, y)
	distances = [distance(click_pos, (bbox.center_x, bbox.center_y)) for bbox in bounding_boxes]
	index = distances.index(min(distances))

	x = bounding_boxes[index].bbox_x1
	y = bounding_boxes[index].bbox_y1
	w = bounding_boxes[index].width
	h = bounding_boxes[index].height
	hw = max(w, h)

	enlarge_factor = 1.1
	x1 = round(max(x + w / 2 - enlarge_factor * hw / 2, 1))
	y1 = round(max(y + h / 2 - enlarge_factor * hw / 2, 1))
	x2 = round(x1 + enlarge_factor * hw)
	y2 = round(y1 + enlarge_factor * hw)

	ret = [(x1, y1)]
	ret.append((x2, y2))
	roi = image[ret[0][1]:ret[1][1], ret[0][0]:ret[1][0], :].copy()
	crop_name = uuid.uuid4().hex + ".png"
	crop_path = os.path.join(IMAGE_UPLOADS, crop_name)
	cv2.imwrite(crop_path, roi)
	print("Saved crop on: ", crop_path)
	return roi, (x1, y1)

def calc_polygon(img):
	newsize = (224, 224)
	im1 = cv2.resize(img, newsize)
	image_np = np.expand_dims(im1, axis=0)
	preds = [model.do_test(polySess, image_np, top_k) for top_k in range(_FIRST_TOP_K)]
	preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)

	#Let's run GGNN now on the bestPoly
	best_poly = preds[0]['polys'][0]
	feature_indexs, poly, mask = utils.preprocess_ggnn_input(best_poly)
	preds_gnn = ggnnModel.do_test(ggnnSess, image_np, feature_indexs, poly, mask)
	refined_poly = preds_gnn['polys_ggnn']

	# Calculate coordinates based on resized ROI
	h, w = im1.shape[:2]
	np_array = np.array(refined_poly[0])
	crop_array = np.array(refined_poly[0])
	crop_array[:, 0] = np_array[:, 0] * w
	crop_array[:, 1] = np_array[:, 1] * h
	
	# Draw and store cropped image with polygon
	fig, axes = plt.subplots(1, num=0,figsize=(12,6))
	axes = np.array(axes).flatten()
	vis_polys(axes[0], im1, crop_array, title='PolygonRNN++')
	fig_name = uuid.uuid4().hex + ".png"
	fig_path = os.path.join(POLYGON_UPLOADS, fig_name)
	fig.savefig(fig_path)

	print("Figure stored in: ", fig_path)

	# Calculate coordinates on original ROI - Without resize
	h, w = img.shape[:2]
	np_array[:, 0] = np_array[:, 0] * w
	np_array[:, 1] = np_array[:, 1] * h

	return np_array, fig_path

def draw_results(img, polygon, coords):
	# Draw polygon on original image
	img = cv2.imread(img.path)[:, :, ::-1]
	polygon[:, 0] = polygon[:, 0] + coords[0]
	polygon[:, 1] = polygon[:, 1] + coords[1]

	fig, axes = plt.subplots(1, num=0,figsize=(36,18))
	axes = np.array(axes).flatten()
	vis_polys(axes[0], img, polygon, title='PolygonRNN++')
	fig_name = uuid.uuid4().hex + ".png"
	fig_path = os.path.join(RESULT_UPLOADS, fig_name)
	fig.savefig(fig_path)
	print("Figure stored in: ", fig_path)
	return fig_path
