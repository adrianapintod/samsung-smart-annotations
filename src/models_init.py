import tensorflow as tf

from models.EvalNet import EvalNet
from models.GGNNPolyModel import GGNNPolygonModel
from models.PolygonModel import PolygonModel

POLYRNN_METAGRAPH='./src/checkpoints/polyrnn/poly/polygonplusplus.ckpt.meta'
POLYRNN_CHECKPOINT='./src/checkpoints/polyrnn/poly/polygonplusplus.ckpt'
EVALNET_CHECKPOINT='./src/checkpoints/polyrnn/evalnet/evalnet.ckpt'
GGNN_METAGRAPH='./src/checkpoints/polyrnn/ggnn/ggnn.ckpt.meta'
GGNN_CHECKPOINT='./src/checkpoints/polyrnn/ggnn/ggnn.ckpt'
_BATCH_SIZE=1

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
