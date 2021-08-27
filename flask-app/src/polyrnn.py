
def polygon_detection(img_path = 'image.png',_BATCH_SIZE = 1, _FIRST_TOP_K = 6 ):


    try:
        #imports
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
    except:
      print("Please install correct version of packages used in this environemnt")

    try:
        #External PATHS
        PolyRNN_metagraph='../models/poly/polygonplusplus.ckpt.meta'
        PolyRNN_checkpoint='../models/poly/polygonplusplus.ckpt'
        EvalNet_checkpoint='../models/evalnet/evalnet.ckpt'
        GGNN_metagraph='../models/ggnn/ggnn.ckpt.meta'
        GGNN_checkpoint='../models/ggnn/ggnn.ckpt'

    except:
      print("Please make sure the pre-trained models are loaded")

    def load_resize_img(img_path = 'image.png'):
        im = Image.open(img_path)
        width, height = im.size
        #print(width, height)
        newsize = (224, 224)
        im1 = im.resize(newsize)
        new_img = 'image_resize.png'
        im1 = im1.save(new_img)
        return new_img

    try:
        # Creating the graphs
        evalGraph = tf.Graph()
        polyGraph = tf.Graph()
        ggnnGraph = tf.Graph()
        #Const
        #_BATCH_SIZE=sys.argv[2]

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
    except:
      print("Unable to initialize models")

    try: 
        #INPUT IMG CROP (224x224x3) -> object should be centered
        #crop_path='image.png'
        crop_path = load_resize_img(img_path)
        print('image loaded successfully')

        #const
        #_FIRST_TOP_K = sys.argv[3]

        #Testing
        image_np = io.imread(crop_path)
        image_np = image_np[:,:,:3]
        image_np = np.expand_dims(image_np, axis=0)
        preds = [model.do_test(polySess, image_np, top_k) for top_k in range(_FIRST_TOP_K)]

        # sort predictions based on the eval score to pick the best.
        preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)
        print('predictions sorted based on eval score')

        #Visualizing TOP_K and scores
        #%matplotlib inline
        print('Visualizing polyrnn model')
        fig, axes = plt.subplots(2,3)
        axes=np.array(axes).flatten()
        [vis_polys(axes[i], image_np[0], np.array(pred['polys'][0]), title='score=%.2f' % pred['scores'][0]) for i,pred in enumerate(preds)]
        plt.show()



        #Let's run GGNN now on the bestPoly
        bestPoly = preds[0]['polys'][0]
        feature_indexs, poly, mask = utils.preprocess_ggnn_input(bestPoly)
        preds_gnn = ggnnModel.do_test(ggnnSess, image_np, feature_indexs, poly, mask)
        refinedPoly=preds_gnn['polys_ggnn']


        #Visualize the final prediction
        print('Visualizing ggnn model')
        fig, ax = plt.subplots(1,1)
        vis_polys(ax,image_np[0],refinedPoly[0], title='PolygonRNN++')
        plt.show()
    except:
      print("Unable to Visualize models")

if __name__=='__main__':
    import warnings
    warnings.filterwarnings("ignore")
    import sys
    polygon_detection(img_path = sys.argv[1],_BATCH_SIZE = int(sys.argv[2]), _FIRST_TOP_K = int(sys.argv[3]))

