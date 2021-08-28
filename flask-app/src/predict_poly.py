import tensorflow as tf
import numpy as np
from PolygonModel import PolygonModel
from EvalNet import EvalNet
from GGNNPolyModel import GGNNPolygonModel
import utils
from poly_utils import vis_polys
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
import sys


def load_model(_BATCH_SIZE = 1):
    PolyRNN_metagraph='../models/poly/polygonplusplus.ckpt.meta'
    PolyRNN_checkpoint='../models/poly/polygonplusplus.ckpt'
    EvalNet_checkpoint='../models/evalnet/evalnet.ckpt'
    GGNN_metagraph='../models/ggnn/ggnn.ckpt.meta'
    GGNN_checkpoint='../models/ggnn/ggnn.ckpt'
    print("Pretrained model is loaded")

    # Creating the graphs
    evalGraph = tf.Graph()
    polyGraph = tf.Graph()
    ggnnGraph = tf.Graph()

    #Initializing and restoring the evaluator net.
    with evalGraph.as_default():
        with tf.variable_scope("discriminator_network"):
            evaluator = EvalNet(_BATCH_SIZE)
            evaluator.build_graph()
        saver = tf.train.Saver()

        # Start session
        evalSess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True
        ), graph=evalGraph)
        saver.restore(evalSess, EvalNet_checkpoint)

    #Initializing and restoring PolyRNN++
    model = PolygonModel(PolyRNN_metagraph, polyGraph)
    model.register_eval_fn(lambda input_: evaluator.do_test(evalSess, input_))
    polySess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True
    ), graph=polyGraph)
    model.saver.restore(polySess, PolyRNN_checkpoint)


    #Initializing and restoring GGNN
    ggnnGraph = tf.Graph()
    ggnnModel = GGNNPolygonModel(GGNN_metagraph, ggnnGraph)
    ggnnSess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True
    ), graph=ggnnGraph)

    ggnnModel.saver.restore(ggnnSess,GGNN_checkpoint)
    print("Model is initialized")
    return model,polySess, ggnnModel,ggnnSess
    


def load_resize_img(img_path = 'image.png'):
    im = Image.open(img_path)
    width, height = im.size
    #print(width, height)
    newsize = (224, 224)
    im1 = im.resize(newsize)
    new_img = 'image_resize.png'
    im1 = im1.save(new_img)
    return new_img


def polygon_detection(model,polySess, ggnnModel,ggnnSess, img_path = 'image.png', _FIRST_TOP_K = 6):

    crop_path = load_resize_img(img_path)
    print('image loaded successfully')

    image_np = io.imread(crop_path)
    image_np = image_np[:,:,:3]
    image_np = np.expand_dims(image_np, axis=0)
    preds = [model.do_test(polySess, image_np, top_k) for top_k in range(_FIRST_TOP_K)]

    # sort predictions based on the eval score to pick the best.
    preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)
    print('predictions sorted based on eval score')

    #Let's run GGNN now on the bestPoly
    bestPoly = preds[0]['polys'][0]
    feature_indexs, poly, mask = utils.preprocess_ggnn_input(bestPoly)
    preds_gnn = ggnnModel.do_test(ggnnSess, image_np, feature_indexs, poly, mask)
    refinedPoly=preds_gnn['polys_ggnn']

    return image_np[0],refinedPoly[0]
