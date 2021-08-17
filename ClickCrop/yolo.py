# import the necessary packages
import os
import time
from math import hypot

import cv2
import numpy as np
# from polyrnn.src.PolygonModel import PolygonModel
# from polyrnn.src.EvalNet import EvalNet
# from polyrnn.src.GGNNPolyModel import GGNNPolygonModel
# import utils
# from poly_utils import vis_polys
import skimage.io as io
import tensorflow as tf

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-y", "--yolo", required=True,
# 	help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
# 	help="threshold when applying non-maxima suppression")
# args = vars(ap.parse_args())

YOLO_PATH = "./ClickCrop/yolo-coco"
LABEL_PATH = "coco.names"
WEIGHTS_PATH = "yolov3.weights"
CONFIG_PATH = "yolov3.cfg"
THRESHOLD = 0.3
CONFIDENCE = 0.5

print(os.getcwd())

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([YOLO_PATH, labels_path])
    LABELS = open(labelsPath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([YOLO_PATH, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([YOLO_PATH, config_path])
    return configPath

def load_model(configPath, weightsPath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net

def load_image(image):
    # load our input image and grab its spatial dimensions
    image = cv2.imread(image)
    # clone = image.copy()
    return image

def get_prediction(image, net, LABELS, COLORS):
	(H, W) = image.shape[:2]
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))


	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	centers = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				centers.append((centerX, centerY))
				confidences.append(float(confidence))
				classIDs.append(classID)


	# # apply non-maxima suppression to suppress weak, overlapping bounding
	# # boxes
	# idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)


	# # ensure at least one detection exists
	# if len(idxs) > 0:
	# 	# loop over the indexes we are keeping
	# 	for i in idxs.flatten():
	# 		# extract the bounding box coordinates
	# 		(x, y) = (boxes[i][0], boxes[i][1])
	# 		(w, h) = (boxes[i][2], boxes[i][3])
	# 		# draw a bounding box rectangle and label on the image
	# 		color = [int(c) for c in COLORS[classIDs[i]]]
	# 		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
	# 		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
	# 		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return (boxes, centers, classIDs, confidences)
    

def get_bounding_boxes(image):
    img = load_image(image)
    labels = get_labels(LABEL_PATH)
    config = get_config(CONFIG_PATH)
    weights = get_weights(WEIGHTS_PATH)
    nets = load_model(config, weights)
    colors = get_colors(labels)
    res = get_prediction(img, nets, labels, colors)
    print(res)
    
