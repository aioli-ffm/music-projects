"""
    Simple Music Classifier: machine learning on music with TensorFlow.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    ############################################################################

    STRUCTURE:
    The file has several parts:
    1. LOAD DATA: get the .wav files as float arrays
    2. SPLIT DATA: functionality to split data in train/test/cv disjoint subsets
    3. FEED DATA: functionality to sample from the subsets to feed the model
    4. ERROR METRICS: functionality for measuring the output of the model
    5. DEFINE TF MODELS: models are actually defined in the models.py file
    6. DEFINE TF GRAPH: a TF superv. class. setup using a model and some config
    7. DEFINE TF SESSION: train and save a model, show results on TensorBoard
    8. RELOAD TF SESSION: reload a pre-trained model and efficiently run it
"""

from __future__ import print_function, division
import numpy as np
import datetime # timestamp for sessions
import os # for listdir and path.join
import random
from six.moves import xrange
from tabulate import tabulate # pretty-print of confusion matrices on terminal
import scipy.io.wavfile as pywav
from scipy import signal # for downsampling
# tensorflow-specific imports
import tensorflow as tf
from tensorboardX import SummaryWriter
# another python file holding the model definitions
import models





################################################################################
# LOAD DATA: files are samplerate=22050, 16-bit wavs
# 10 classes, 100 unique 30s wavs per class
# reggae, classical, country, jazz, metal, pop, disco, hiphop, rock, blues
################################################################################

def normalize_array(nparr):
    peak = max(abs(nparr.max()), abs(nparr.min()))
    return nparr.astype(np.float32)/peak

def get_dataset(dataset_path, downsample=0, normalize=True):
    """Returns GTZAN dataset as a dictionary of key='classname',
       value='wavlist', where wavlist[XX] corresponds to
       'dataset_path/classname/classname.000XX.wav' as numpy array
       :param downsample: int By which factor the audio files are downsampled.
        if this is 0, the data won't be downsampled.
    """
    data = {c:[] for c in os.listdir(dataset_path)}
    # fill the data dictionary: "DATASET_PATH/classname/classname.000XX.wav"
    for classname, wavlist in data.iteritems():
        class_path = os.path.join(dataset_path, classname)
        for i in xrange(100):
            wav_path = os.path.join(class_path, classname+"."+str(i).zfill(5))
            arr = pywav.read(wav_path+".wav")[1].astype(np.float32)
            if normalize: arr /= max(abs(arr.max()), abs(arr.min()))
            if downsample > 0:
                arr = signal.decimate(arr, downsample).astype(np.float32)
            wavlist.append(arr)
    return data


################################################################################
# SPLIT DATA:
################################################################################

def split_dataset(dataset, train_cv_test_ratio, classes=None):
    """Given a dictionary of key=class, value=wav_list, shuffles every wav_list
       for each class and splits it by the proportions given by the
       [train, cv, test] ratio list (which must add up to 1), and returns the 3
       subsets as 3 dicts in that order. If a class list is given, only the
       subsets for that classes will be returned.
    """
    if sum(train_cv_test_ratio)!=1:
        raise RuntimeError("[ERROR] split_dataset: ratios don't add up to 1! ")
    train_subset = {}
    cv_subset = {}
    test_subset = {}
    classes = classes if classes else dataset.keys()
    for classname in classes:
        wavlist = dataset[classname]
        random.shuffle(wavlist)
        # get min and max indexes as given by the ratios
        l = len(wavlist) # always 100 for GTZAN
        cv_0 = int(l*train_cv_test_ratio[0])
        test_0 = cv_0+int(l*train_cv_test_ratio[1])
        # feed the subsets
        train_subset[classname] = wavlist[0:cv_0]
        cv_subset[classname] = wavlist[cv_0:test_0]
        test_subset[classname] = wavlist[test_0:]
    return train_subset, cv_subset, test_subset



################################################################################
# FEED DATA: random for training, per-class for CV and test
################################################################################

def get_random_batch(dataset, chunk_size, batch_size):
    """Given a dictionary of key=class, value=wav_list, returns a rank 2 tensor
       of batch_size elements of length chunk_size, sampled from randomized
       classes and positions. To allow supervised learning, it also returns the
       classes in the corresponding order.
       Due to its randomization (and low efficiency), it is intended to be
       used to sample batches from a small, balanced training set, e.g:
         data_batch, labels_batch = get_random_batch(TRAIN_SUBSET, 100, 12)
    """
    # not very efficient algo but OK in this setup:
    # 1. get BATCH_SIZE random labels, and from each label a respective wav ID
    labels = [random.choice(dataset.keys()) for _ in xrange(batch_size)]
    max_per_class = {cl : len(wavlist)-1 for cl, wavlist in dataset.iteritems()}
    # if CHUNK_SIZE<wav_len, the exact chunk position is also randomized:
    wav_ids = [random.randint(0, max_per_class[l]) for l in labels]
    lengths = [dataset[labels[x]][wav_ids[x]].shape[0]
               for x in xrange(batch_size)]
    start_idxs = [random.randint(0, lengths[x]-chunk_size)
                  for x in xrange(batch_size)]
    # now that we know class, id and start_idx for each chunk, collect tensor:
    data = np.stack([dataset[labels[x]][wav_ids[x]][start_idxs[x]:
                                                    start_idxs[x]+chunk_size]
                     for x in xrange(batch_size)])
    return data, labels

def get_class_batch(dataset,clss, chunk_size):
    """Given adictionary of key=class, value=wav_list, returns a rank 2 tensor
       with the most possible non-overlapping distinct chunks of chunk_size
       from the requested class. For example, if a class has 10 wavs of size 9,
       and the chunk_size is 2, this function will return 80 chunks: the
       remaining '1' of each wav is discarded.

       Note that this doesn't have a batch size, since performing validation
       or testing usually involves taking every sample into account once.
       Having all the chunks of the same class in a single batch also allows
       to perform voting on them and other metrics that don't apply to training.
       Usage example:
       for cl in CV_SUBSET:
           x = get_class_batch(CV_SUBSET, cl, 1000)
           print cl + "  "+ str(x.shape)
    """
    wav_list = dataset[clss]
    # for each wav, truncate the end if shorter than chunk_size and split
    wav_chunks = np.stack([w[x:x+chunk_size] for w in wav_list
                           for x in xrange(0,
                                           w.shape[0]-(w.shape[0]%chunk_size),
                                           chunk_size)])
    return wav_chunks





################################################################################
# ERROR METRICS
################################################################################

class ConfusionMatrix(object):
    """
    """
    def __init__(self, class_domain, name=""):
        self.matrix = {c:{c:0 for c in class_domain} for c in class_domain}
        self.name = name
    def add(self, predictions, labels):
        """Given a list of predictions and their respective labels, adds every
           corresponding entry to the confusion matrix.
        """
        pred_lbl = zip(predictions, labels)
        for pred, lbl in pred_lbl:
            self.matrix[lbl][pred] += 1
    def __str__(self):
        acc, acc_by_class = self.accuracy()
        classes = sorted(self.matrix.keys())
        short_classes = {c: c[0:8]+"..." if len(c)>8 else c for c in classes}
        prettymatrix = tabulate([[short_classes[c1]]+[self.matrix[c1][c2]
                                                      for c2 in classes]+
                                 [acc_by_class[c1]]
                                 for c1 in classes],
                                headers=["real(row)|predicted(col)"]+
                                [short_classes[c] for c in classes]+
                                ["acc. by class"])
        return ("\n"+self.name+" CONFUSION MATRIX\n"+prettymatrix+
                "\n"+self.name+" ACCURACY="+str(acc)+"\n")
    def accuracy(self):
        """Returns the total accuracy, and a dict with the accuracy per class
        """
        total = 0
        right = 0
        by_class = {c: [0,0] for c in self.matrix}
        acc = float("nan")
        by_class_acc = {c:float("nan") for c in self.matrix}
        for clss, preds in self.matrix.iteritems():
            for pred, n in preds.iteritems():
                total += n
                by_class[clss][1] += n
                if clss==pred:
                    right += n
                    by_class[clss][0] += n
        try:
            acc = float(right)/total
        except ZeroDivisionError:
            pass
        for c,x in by_class.iteritems():
            try:
                by_class_acc[c] = float(x[0])/x[1]
            except ZeroDivisionError:
                pass
        return acc, by_class_acc





################################################################################
# DEFINE TENSORFLOW MODELS
################################################################################

# see the models.py file



################################################################################
# DEFINE TENSORFLOW GRAPH
################################################################################

softmax = tf.nn.sparse_softmax_cross_entropy_with_logits
signature = tf.saved_model.signature_def_utils.predict_signature_def

def make_graph(model, chunk_shape, num_classes, l2reg=0,
               optimizer_fn=lambda:tf.train.AdamOptimizer()):
    with tf.Graph().as_default() as g:
        data_ph = tf.placeholder(tf.float32, shape=((None,)+chunk_shape),
                                 name="data_placeholder")
        labels_ph = tf.placeholder(tf.int32, shape=(None),
                                   name="labels_placeholder")
        logits, l2nodes = model(data_ph, num_classes)
        loss = tf.reduce_mean(softmax(logits=logits, labels=labels_ph))
        if l2reg>0:
            loss += l2reg*l2nodes
        global_step = tf.Variable(0, name="global_step", trainable=False)
        minimizer = optimizer_fn().minimize(loss, global_step=global_step)
        predictions = tf.argmax(logits, 1, output_type=tf.int32, name="preds")
        inputs = [data_ph, labels_ph]
        outputs = [logits, loss, global_step, minimizer, predictions]
        return g, inputs, outputs



################################################################################
# DEFINE TF SESSION
################################################################################

def make_timestamp():
    return '{:%d_%b_%Y_%Hh%Mm%Ss}'.format(datetime.datetime.now())

def make_session_datastring(model, optimizer_fn, batch_size, chunk_size, l2_reg,  extra_info="", separator="||"):
    s = str(separator)
    if extra_info:
        extra_info = s+extra_info
    return (make_timestamp()+"/"+model.__name__+s+str(opt_manager)+s+"batchsize_"+
            str(batch_size)+s+"chunksize_"+str(chunk_size)+s+"L2_"+str(l2_0)+s+"dropout_"+
            str(dropout_0)+s+"cvvotingsize_"+str(cv_voting_size)+s+"testvotingsize_"+
            str(test_voting_size)+s+"normalizechunks_"+str(normalize_chunks)+extra_info)


def run_training(train_subset, cv_subset, test_subset, model,
                 batch_size, chunk_size, max_steps=float("inf"),
                 l2reg=0, optimizer_fn=lambda: tf.train.AdamOptimizer(),
                 train_freq=10, cv_freq=100,
                 save_path=None):
    # get current classes and map them to 0,1,2,3... integers
    # classes = {c:i for i, c in enumerate(train_subset.keys())}
    # classes_inv = {v:k for k,v in classes.iteritems()}
    #  LOGGER (to plot them to TENSORBOARD)
    logger = SummaryWriter()
    logger.add_text("Session info", make_timestamp() +
                    " model=%s batchsz=%d chunksz=%d l2reg=%f" %
                    (model.__name__, batch_size, chunk_size, l2reg), 0)
    # CREATE TF GRAPH
    graph,[data_ph,labels_ph],[logits,loss,global_step,minimizer,predictions]=(
        make_graph(model, (chunk_size,), len(train_subset), l2reg,optimizer_fn))
    # CREATE AND CONFIGURE GRAPH SAVER:
    out_path = os.path.join("./saved_models", make_timestamp())
    if save_path:
        saver = tf.saved_model.builder.SavedModelBuilder(save_path)
    # START SESSION (log_device_placement=True)
    with tf.Session(graph=graph, config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        # START TRAINING
        for step in xrange(max_steps):
            data_batch, label_batch = get_random_batch(train_subset, chunk_size,
                                                       batch_size)
            lbl = [CLASS2INT[l] for l in label_batch] # convert from str to int
            sess.run(minimizer, feed_dict={data_ph:data_batch, labels_ph:lbl})
            # LOG TRAINING
            if(step%train_freq==0):
                p, l, i, logts = sess.run([predictions,loss,global_step,logits],
                                feed_dict={data_ph:data_batch, labels_ph:lbl})
                cm_train = ConfusionMatrix(train_subset.keys(), "BATCH")
                cm_train.add([INT2CLASS[x] for x in p], label_batch)
                acc, by_class_acc = cm_train.accuracy()
                print("step:%d"%i, cm_train)
                logger.add_scalars("TRAIN", {"acc":acc, "avg_loss":l} ,i)
            # LOG CROSS-VALIDATION
            if(step%cv_freq==0):
                cv_loss = 0
                cm_cv = ConfusionMatrix(train_subset.keys(), "CV")
                for c_cv in train_subset:
                    print("validating class %s: this may take a while..."%c_cv)
                    # extract ALL chunks of selected class and corresponding lbl
                    cv_data = get_class_batch(cv_subset, c_cv, chunk_size)
                    cv_labels = [c_cv for _ in xrange(len(cv_data))]
                    lbl = [CLASS2INT[c_cv] for _ in xrange(len(cv_data))]
                    p,l,i = sess.run([predictions, loss, global_step],
                                     feed_dict={data_ph:cv_data, labels_ph:lbl})
                    cv_loss += l
                    cm_cv.add([INT2CLASS[x] for x in p], cv_labels)
                # once loss and matrix has been calculated for every class...
                acc, by_class_acc = cm_cv.accuracy()
                print("step:%d"%i, cm_cv)
                logger.add_scalars("CV", {"acc":acc,
                                          "avg_loss":cv_loss/len(cv_subset)}, i)
        # AFTER TRAINING LOOP ENDS, DO VALIDATION ON THE TEST SUBSET (omitted
        # here for brevity, code is identical to the one for cross-validation)
        print("here could be your amazing test with 99.9% accuracy!!")
        # save graph
        if save_path:
            graph_signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={"data":data_ph, "classes":labels_ph}, outputs={"logits":logits,"loss":loss,"global_step":global_step, "predictions":predictions})
            saver.add_meta_graph_and_variables(sess, ["my_model"],
                                               signature_def_map={"my_signature":graph_signature})
            saver.save()
            print("trained model saved to", save_path)



################################################################################
# RELOAD TF SESSION
################################################################################

class TrainedModel(object):
    def __init__(self, chunk_size, savepath, model_name="my_model"):
        self.chunk_size = chunk_size
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)
        tf.saved_model.loader.load(self.sess, [model_name], savepath)
        self.data_ph = self.g.get_tensor_by_name("data_placeholder:0")
        self.predictions = self.g.get_tensor_by_name("preds:0")
    def run(self, data_batch):
        return
    def run_and_eval(self, data_subset, batch_size):
        data_batch, label_batch = get_random_batch(data_subset,
                                                   self.chunk_size, batch_size)
        argmx = self.sess.run(self.predictions,feed_dict={self.data_ph:data_batch})
        cm = ConfusionMatrix(data_subset.keys(), "RELOADED")
        cm.add([INT2CLASS[x] for x in argmx], label_batch)
        return cm
    def close(self):
        self.sess.close()




################################################################################
# RUN TF SESSION
################################################################################

def load_mnist_as_ddicts():
    """
    """
    # load the dataset from TF dependencies
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # extract the tensors and print their shape
    train_labels = mnist.train._labels # 55000
    train_images = mnist.train._images.reshape((-1, 28*28))
    cv_labels = mnist.validation._labels # 5000
    cv_images = mnist.validation._images.reshape((-1, 28*28))
    test_labels = mnist.test._labels # 10000
    test_images = mnist.test._images.reshape((-1, 28*28))
    print(train_labels.shape, train_images.shape, cv_labels.shape, cv_images.shape,
          test_labels.shape, test_images.shape)
    # store them as dicts of lists
    train_dict = {str(i):[] for i in xrange(10)} # this are still empty!
    cv_dict = {str(i):[] for i in xrange(10)}
    test_dict = {str(i):[] for i in xrange(10)}
    for i in xrange(len(train_labels)):
        train_dict[str(train_labels[i])].append(train_images[i])
    for i in xrange(len(cv_labels)):
        cv_dict[str(cv_labels[i])].append(cv_images[i])
    for i in xrange(len(test_labels)):
        test_dict[str(test_labels[i])].append(test_images[i])
    # return them
    return train_dict, cv_dict, test_dict


def test_with_mnist():
    classes = [str(x) for x in range(10)]
    model = lambda batch, num_classes: simple_mlp(batch, num_classes, 128)
    batch_size = 500
    chunk_size = 784
    max_steps = 1001
    l2_reg = 1e-7
    optimizer_fn = lambda: tf.train.AdamOptimizer(1e-3)
    train_freq = 50
    cv_freq = 1000
    train_subset, cv_subset, test_subset = load_mnist_as_ddicts()
    #
    run_training(train_subset, cv_subset, test_subset, model,
                 batch_size, chunk_size, max_steps,
                 l2_reg, optimizer_fn,
                 train_freq, cv_freq)


# test_with_mnist()


# GLOBALS AND HYPERPARAMETERS ##################################################
DATASET_PATH = "../../datasets/gtzan/"
TRAIN_CV_TEST_RATIO = [0.7, 0.1, 0.2]
CLASSES =  ["reggae", "classical", "country", "jazz", "metal", "pop", "disco", "hiphop", "rock", "blues"]
CLASS2INT = {"reggae":0, "classical":1, "country":2, "jazz":3, "metal":4, "pop":5, "disco":6, "hiphop":7, "rock":8, "blues":9}
INT2CLASS = {v:k for k,v in CLASS2INT.iteritems()}
MODEL=  models.basic_convnet
# lambda batch, num_classes: models.deeper_mlp(batch, num_classes, 512, 64) #simple_mlp(batch, num_classes, 1000)
DOWNSAMPLE=7
BATCH_SIZE = 1000
CHUNK_SIZE = (22050*2)//DOWNSAMPLE
MAX_STEPS=2001
L2_REG = 1e-3
OPTIMIZER_FN = lambda: tf.train.AdamOptimizer(1e-3)
TRAIN_FREQ=10
CV_FREQ=100
################################################################################


DATA = get_dataset(DATASET_PATH, downsample=DOWNSAMPLE, normalize=True)
TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET = split_dataset(DATA, TRAIN_CV_TEST_RATIO,
                                                     CLASSES)
# CV_SUBSET = {k:v[0:2] for k,v in CV_SUBSET.iteritems()}
del DATA # data won't be needed anymore and may free some useful RAM





savepath = os.path.join("./saved_models", make_timestamp())
run_training(TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET, MODEL,
             BATCH_SIZE, CHUNK_SIZE, MAX_STEPS,
             L2_REG, OPTIMIZER_FN,
             TRAIN_FREQ, CV_FREQ,
             save_path=savepath)



print("\n\n\n\n\n\n\n\n\n")

# # saved_model_cli show --dir saved_models/28_Nov_2017_16h43m53s/ --tag_set my_model --signature_def model

# g = tf.Graph()
# with tf.Session(graph=g) as sess:
#     tf.saved_model.loader.load(sess, ["my_model"], savepath)
#     data_ph = g.get_tensor_by_name("data_placeholder:0")
#     k = g.get_tensor_by_name("ArgMax:0")
#     for i in xrange(5):
#         data_batch, label_batch = get_random_batch(TRAIN_SUBSET, CHUNK_SIZE,
#                                                    BATCH_SIZE)
#         lbl = [CLASS2INT[l] for l in label_batch] # convert from str to int
#         argmx = sess.run(k, feed_dict={data_ph:data_batch})
#         cm_train = ConfusionMatrix(TRAIN_SUBSET.keys(), "RELOADED")
#         cm_train.add([INT2CLASS[x] for x in argmx], label_batch)
#         acc, by_class_acc = cm_train.accuracy()
#         print("step:%d"%i, cm_train)


m  = TrainedModel(CHUNK_SIZE, savepath)
for i in xrange(10000):
    if (i%1000==0):
        m.run_and_eval(TRAIN_SUBSET, 1000)
        print("quack!", i)

m.close()
