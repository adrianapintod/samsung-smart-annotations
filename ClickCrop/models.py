import tensorflow as tf

from EvalNet import EvalNet
from GGNNPolyModel import GGNNPolygonModel
from PolygonModel import PolygonModel

POLYRNN_METAGRAPH='./ClickCrop/polyrnn/models/poly/polygonplusplus.ckpt.meta'
POLYRNN_CHECKPOINT='./ClickCrop/polyrnn/models/poly/polygonplusplus.ckpt'
EVALNET_CHECKPOINT='./ClickCrop/polyrnn/models/evalnet/evalnet.ckpt'
GGNN_METAGRAPH='./ClickCrop/polyrnn/models/ggnn/ggnn.ckpt.meta'
GGNN_CHECKPOINT='./ClickCrop/polyrnn/models/ggnn/ggnn.ckpt'
_BATCH_SIZE=1
_FIRST_TOP_K = 6

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
    saver.restore(evalSess, EVALNET_CHECKPOINT)

#Initializing and restoring PolyRNN++
model = PolygonModel(POLYRNN_METAGRAPH, polyGraph)
model.register_eval_fn(lambda input_: evaluator.do_test(evalSess, input_))
polySess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True
), graph=polyGraph)
model.saver.restore(polySess, POLYRNN_CHECKPOINT)

#Initializing and restoring GGNN
ggnnGraph = tf.Graph()
ggnnModel = GGNNPolygonModel(GGNN_METAGRAPH, ggnnGraph)
ggnnSess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True
), graph=ggnnGraph)

ggnnModel.saver.restore(ggnnSess,GGNN_CHECKPOINT)
