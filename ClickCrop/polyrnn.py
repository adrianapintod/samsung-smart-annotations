import argparse
import os
import time
from math import hypot

import cv2
import numpy as np
import skimage.io as io
import tensorflow as tf

import utils
from poly_utils import vis_polys
from EvalNet import EvalNet
from GGNNPolyModel import GGNNPolygonModel
from PolygonModel import PolygonModel
from entities import BoundingBox
from models import model, ggnnModel, ggnnSess, polySess


IMAGE_UPLOADS = "./imgs"
POLYRNN_METAGRAPH='./ClickCrop/polyrnn/models/poly/polygonplusplus.ckpt.meta'
POLYRNN_CHECKPOINT='./ClickCrop/polyrnn/models/poly/polygonplusplus.ckpt'
EVALNET_CHECKPOINT='./ClickCrop/polyrnn/models/evalnet/evalnet.ckpt'
GGNN_METAGRAPH='./ClickCrop/polyrnn/models/ggnn/ggnn.ckpt.meta'
GGNN_CHECKPOINT='./ClickCrop/polyrnn/models/ggnn/ggnn.ckpt'

def distance(p1, p2):
#"""Euclidean distance between two points."""
	# print(p2)
	xx1, yy1 = p1
	xx2, yy2 = p2
	return hypot(xx2 - xx1, yy2 - yy1)

def select_bbox(img, x, y):
	image = cv2.imread(img.path)
	bounding_boxes = BoundingBox.query.filter_by(image_id=img.id).all()
	click_pos = (x, y)
	distances = [distance(click_pos, (bbox.center_x, bbox.center_y)) for bbox in bounding_boxes]
	index = distances.index(min(distances))

	x = bounding_boxes[index].bbox_x1
	y = bounding_boxes[index].bbox_y1
	w = bounding_boxes[index].width
	h = bounding_boxes[index].height
	hw = max(w, h)

	# x1 = bounding_boxes[index].bbox_x1
	# y1 = bounding_boxes[index].bbox_y1
	# x2 = bounding_boxes[index].bbox_x2
	# y2 = bounding_boxes[index].bbox_y2

	enlarge_factor = 1.1
	x1 = round(max(x + w / 2 - enlarge_factor * hw / 2, 1))
	y1 = round(max(y + h / 2 - enlarge_factor * hw / 2, 1))
	x2 = round(x1 + enlarge_factor * hw)
	y2 = round(y1 + enlarge_factor * hw)

	ret = [(x1, y1)]
	ret.append((x2, y2))
	roi = image[ret[0][1]:ret[1][1], ret[0][0]:ret[1][0], :].copy()
	cv2.imwrite(os.path.join(IMAGE_UPLOADS, "CroppedImage.png"), roi)
	return roi

def calc_polygon(img):
	# _BATCH_SIZE=1
	_FIRST_TOP_K = 6

	# # Creating the graphs
	# evalGraph = tf.Graph()
	# polyGraph = tf.Graph()
	# ggnnGraph = tf.Graph()

	# #Initializing and restoring the evaluator net.
	# with evalGraph.as_default():
	# 	with tf.variable_scope("discriminator_network"):
	# 		evaluator = EvalNet(_BATCH_SIZE)
	# 		evaluator.build_graph()
	# 	saver = tf.train.Saver()

	# 	# Start session
	# 	evalSess = tf.Session(config=tf.ConfigProto(
	# 		allow_soft_placement=True
	# 	), graph=evalGraph)
	# 	saver.restore(evalSess, EVALNET_CHECKPOINT)

	# #Initializing and restoring PolyRNN++
	# model = PolygonModel(POLYRNN_METAGRAPH, polyGraph)
	# model.register_eval_fn(lambda input_: evaluator.do_test(evalSess, input_))
	# polySess = tf.Session(config=tf.ConfigProto(
	# 	allow_soft_placement=True
	# ), graph=polyGraph)
	# model.saver.restore(polySess, POLYRNN_CHECKPOINT)

	# #Initializing and restoring GGNN
	# ggnnGraph = tf.Graph()
	# ggnnModel = GGNNPolygonModel(GGNN_METAGRAPH, ggnnGraph)
	# ggnnSess = tf.Session(config=tf.ConfigProto(
	# 	allow_soft_placement=True
	# ), graph=ggnnGraph)

	# ggnnModel.saver.restore(ggnnSess,GGNN_CHECKPOINT)

	# print(img.shape)
	newsize = (224, 224)
	im1 = cv2.resize(img, newsize)
	image_np = np.expand_dims(im1, axis=0)
	preds = [model.do_test(polySess, image_np, top_k) for top_k in range(_FIRST_TOP_K)]
	preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)

	#Let's run GGNN now on the bestPoly
	bestPoly = preds[0]['polys'][0]
	feature_indexs, poly, mask = utils.preprocess_ggnn_input(bestPoly)
	preds_gnn = ggnnModel.do_test(ggnnSess, image_np, feature_indexs, poly, mask)
	refinedPoly = preds_gnn['polys_ggnn']
	return refinedPoly
