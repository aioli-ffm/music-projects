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
    Following a 'notebook' approach, this file holds the complete pipeline in a
    sequence:
    1. LOAD DATA: get the .wav files as float arrays
    2. SPLIT DATA: functionality to split data in train/test/cv disjoint subsets
    3. FEED DATA: functionality to sample from the subsets to feed the model
    4. ERROR METRICS: functionality for measuring the output of the model
    5. DEFINE TF MODELS: models are actually defined in the models.py file
    6. DEFINE TF GRAPH: a TF superv. class. setup using a model and some config
    7. TRAINING TF SESSION: train and save a model, show results on TensorBoard
    8. RELOADING TF SESSION: reload a pre-trained model and efficiently run it
    9. RUN ALL: load data, train+eval+save+reload model, show in TensorBoard
"""


from __future__ import print_function, division
import numpy as np
import datetime # timestamp for sessions
from time import time # rough time measurements
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
    """Given a NumPy array, returns it as float32 with values between -1 and 1
    """
    peak = max(abs(nparr.max()), abs(nparr.min()))
    return nparr.astype(np.float32)/peak

def get_dataset(dataset_path, downsample_ratio=0):
    """Returns GTZAN dataset as a dictionary of key='classname',
       value='wavlist' where wavlist is a python list of np.float32 arrays.
       If normalize=True
       with values between -1 and 1, and wavlist[23]corresponds to
       'dataset_path/classname/classname.00023.wav'.
       :param downsample_ratio: is a positive integer by which factor the audio
       files are downsampled. If 0 or 1, the data won't be downsampled.
    """
    data = {c:[] for c in os.listdir(dataset_path)}
    # fill the data dictionary: "DATASET_PATH/classname/classname.000XX.wav"
    for classname, wavlist in data.iteritems():
        class_path = os.path.join(dataset_path, classname)
        for i in xrange(100):
            wav_path = os.path.join(class_path, classname+"."+str(i).zfill(5))
            arr = pywav.read(wav_path+".wav")[1].astype(np.float32)
            arr /= max(abs(arr.max()), abs(arr.min()))
            # decimate is a scipy function that downsamples the array
            if downsample_ratio > 0:
                arr = signal.decimate(arr, downsample_ratio).astype(np.float32)
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
    """Minimal implementation of a confusion matrix as dict of dicts, with
       basic functionality to add_batch, pretty-print and get overall and
       per-class accuracy. Usage example:
        cm = ConfusionMatrix(["classical", "swing", "reggae"])
        cm.add(["swing", "swing", "swing"], ["classical", "reggae", "swing"])
        print(cm)
    """

    def __init__(self, class_domain, name=""):
        """Class domain is any iterable holding hashable items (usually a list
           of strings), which will serve as indexes for the add method and as
           labels for the pretty-printing. The name is optional also for pretty-
           printing purposes.
        """
        self.matrix = {c:{c:0 for c in class_domain} for c in class_domain}
        self.name = name

    def add(self, predictions, labels):
        """Given a list of predictions and their respective labels, adds every
           corresponding entry to the confusion matrix. Note that the contents
           of both lists have to be members of the class_domain iterable given
           when
        """
        pred_lbl = zip(predictions, labels)
        for pred, lbl in pred_lbl:
            self.matrix[lbl][pred] += 1

    def __str__(self):
        """Pretty-print for the confusion matrix, showing the title, the
           contents and the accuracy (overall and by-class).
        """
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
        """Returns the total accuracy, and a dict with {class : by-class-acc}.
        """
        total = 0    # accuracy is right/total
        diagonal = 0 # "right" entries are in the diagonal
        by_class = {c: [0,0] for c in self.matrix}
        acc = float("nan")
        by_class_acc = {c:float("nan") for c in self.matrix}
        for clss, preds in self.matrix.iteritems():
            for pred, n in preds.iteritems():
                total += n
                by_class[clss][1] += n
                if clss==pred:
                    diagonal += n
                    by_class[clss][0] += n
        # if some "total" was zero, acc is NaN. Else calculate diagonal/total
        try:
            acc = float(diagonal)/total
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

def make_graph(model, chunk_shape, num_classes, l2reg=0,
               optimizer_fn=lambda:tf.train.AdamOptimizer()):
    """Every supervised learning setup has common requirements. This function
       implements them, leaving the input parameters modular:
        * model: a python function. It will be called like this:
          logits, l2nodes = model(data_ph, num_classes) so it must fulfill this
          interface: accept a tf.float32 placeholder and an integer, and return
          the logits and the l2loss nodes. See the models.py file for examples
        * chunk_shape: a tuple. For rank 1 tensors of length X, it is (X,).
        * l2reg: a positive float being the L2 regularization factor. Note that
          it is added to the loss without averaging, e.g. loss+=l2reg*l2nodes.
        * optimizer_fn: the wanted optimizer wrapped in a lambda, e.g.:
          lambda: tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.5)
    """
    with tf.Graph().as_default() as g:
        # GRAPH INPUTS
        data_ph = tf.placeholder(tf.float32, shape=((None,)+chunk_shape),
                                 name="data_placeholder")
        labels_ph = tf.placeholder(tf.int32, shape=(None),
                                   name="labels_placeholder")
        # EMBEDDING MODEL IN GRAPH
        logits, l2nodes = model(data_ph, num_classes)
        predictions = tf.argmax(logits, 1, output_type=tf.int32, name="preds")
        # COST FUNCTION AND OPTIMIZER ON TOP OF MODEL
        loss = tf.reduce_mean(softmax(logits=logits, labels=labels_ph))
        if l2reg>0:
            loss += l2reg*l2nodes
        global_step = tf.Variable(0, name="global_step", trainable=False)
        minimizer = optimizer_fn().minimize(loss, global_step=global_step)
        # RETURN GRAPH AND ITS IN/OUT SIGNATURE
        inputs = [data_ph, labels_ph]
        outputs = [logits, loss, global_step, minimizer, predictions]
        return g, inputs, outputs



################################################################################
# TRAINING TF SESSION
################################################################################

signature = tf.saved_model.signature_def_utils.predict_signature_def

def make_timestamp():
    """Sample output: 01_Jan_2019_12h00m05s
    """
    return '{:%d_%b_%Y_%Hh%Mm%Ss}'.format(datetime.datetime.now())

def run_training(train_subset, cv_subset, test_subset, model,
                 batch_size, chunk_size, max_steps=float("inf"),
                 l2reg=0, optimizer_fn=lambda: tf.train.AdamOptimizer(),
                 train_freq=10, cv_freq=100,
                 save_path=None):
    """This function is kept within the scope of a single 'with' tf.Session
       context for (believe it or not!) the sake of readability. Given the
       (hopefully) self-explained parameter list, it performs the following
       sequence:
        1. Instantiate the TensorBoard logger and TF graph
        2. Start the TF session based on the instantiated TF graph and init vars
           3. Training loop: for every step, load a random batch and train on it
              4. Every train_freq steps, log the results and error metrics on
                 the given training batch
              5. Every cv_freq steps, log results and error metrics on the whole
                 validation subset
           6. After training ends, same as 5. but for test subset (not impl.)
           7. Save the graph and variable values to save_path
    """
    #  LOGGER (for the interaction with TensorBoard)
    logger = SummaryWriter()
    logger.add_text("Session info", make_timestamp() +
                    " model=%s batchsz=%d chunksz=%d l2reg=%f" %
                    (model.__name__, batch_size, chunk_size, l2reg), 0)
    # 1. CREATE TF GRAPH
    graph,[data_ph,labels_ph],[logits,loss,global_step,minimizer,predictions]=(
        make_graph(model, (chunk_size,), len(train_subset), l2reg,optimizer_fn))
    # 2. START SESSION (log_device_placement=True)
    with tf.Session(graph=graph, config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        # 3. TRAINING
        for step in xrange(max_steps):
            data_batch, label_batch = get_random_batch(train_subset, chunk_size,
                                                       batch_size)
            lbl = [CLASS2INT[l] for l in label_batch] # convert from str to int
            sess.run(minimizer, feed_dict={data_ph:data_batch, labels_ph:lbl})
            # 4. LOG TRAINING
            if(step%train_freq==0):
                p, l, i, logts = sess.run([predictions,loss,global_step,logits],
                                feed_dict={data_ph:data_batch, labels_ph:lbl})
                cm_train = ConfusionMatrix(train_subset.keys(), "BATCH")
                cm_train.add([INT2CLASS[x] for x in p], label_batch)
                acc, by_class_acc = cm_train.accuracy()
                print("step:%d"%i, cm_train)
                logger.add_scalars("TRAIN", {"acc":acc, "avg_loss":l} ,i)
            # 5. LOG CROSS-VALIDATION
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
        # 6. AFTER TRAINING LOOP ENDS, DO VALIDATION ON THE TEST SUBSET (omitted
        # here for brevity, code is identical to the one for cross-validation)
        print("here could be your amazing test with 99.9% accuracy!!")
        # 7. SAVE GRAPH STRUCTURE AND VARIABLES
        if save_path:
            out_path = os.path.join("./saved_models", make_timestamp())
            saver = tf.saved_model.builder.SavedModelBuilder(save_path)
            graph_sig = signature(inputs={"data":data_ph, "classes":labels_ph},
                                  outputs={"logits":logits,"loss":loss,
                                           "global_step":global_step,
                                           "predictions":predictions})
            saver.add_meta_graph_and_variables(sess, ["my_model"],
                                               signature_def_map={"my_signature"
                                                                  :graph_sig})
            saver.save()
            print("trained model saved to", save_path)



################################################################################
# RELOADING TF SESSION
################################################################################

class TrainedModel(object):
    """This class can load a model that has been pre-trained and saved by the
       run_training function. For that, it has been passed the same chunk_size
       and savepath parameters. Once Instantiated, the model can be efficiently
       run on batches of shape (X, chunk_size) for arbitrary X using
       run_and_eval. The class implements the context interface, so it can be
       used like this:
         with TrainedModel(CHUNK_SIZE, savepath) as m:
          for i in xrange(100000):
              if (i%1000==0):
                  m.run_and_eval(TRAIN_SUBSET, 1000)
                  print("model ran", i, "times")
       Alternatively:
         m = TrainedModel(CHUNK_SIZE, savepath)
         for ...
         m.close()
    """
    def __init__(self, chunk_size, savepath, model_name="my_model"):
        """Given a path to a pre-saved model, and the chunk size that the
           model accepts, loads the model into a new TF graph, so it can
           be used with the run_and_eval method
        """
        self.chunk_size = chunk_size
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)
        tf.saved_model.loader.load(self.sess, [model_name], savepath)
        self.data_ph = self.g.get_tensor_by_name("data_placeholder:0")
        self.predictions = self.g.get_tensor_by_name("preds:0")

    def run(self, data_batch):
        """Runs the model for the given data batch. Note that the chunk size
           has to be accepted by the model.
        """
        return self.sess.run(self.predictions, feed_dict={self.data_ph:data_batch})

    def close(self):
        self.sess.close()

    # minimal implementation for the context manager interface
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
            return False
        self.close() # the single purpose for the context manager is this



################################################################################
# RUN ALL
################################################################################

# GLOBALS AND HYPERPARAMETERS
DATASET_PATH = "../../datasets/gtzan/"
TRAIN_CV_TEST_RATIO = [0.7, 0.1, 0.2]
CLASSES =  ["reggae", "classical", "country", "jazz", "metal", "pop", "disco",
            "hiphop", "rock", "blues"]
CLASS2INT = {"reggae":0, "classical":1, "country":2, "jazz":3, "metal":4,
             "pop":5, "disco":6, "hiphop":7, "rock":8, "blues":9}
INT2CLASS = {v:k for k,v in CLASS2INT.iteritems()}
MODEL=  models.simple_mlp # basic_convnet
# lambda batch, num_classes: models.deeper_mlp(batch, num_classes, 512, 64)
#simple_mlp(batch, num_classes, 1000)
DOWNSAMPLE= 7
BATCH_SIZE = 1000
CHUNK_SIZE = (22050*2)//DOWNSAMPLE
MAX_STEPS=1001
L2_REG = 1e-3
OPTIMIZER_FN = lambda: tf.train.AdamOptimizer(1e-3)
TRAIN_FREQ=10
CV_FREQ=100

def run_pretrained_model(data_subset, savepath, chunk_size, batch_size, iters):
    """
    """
    with TrainedModel(chunk_size, savepath) as m:
        cm = ConfusionMatrix(data_subset.keys(), "RELOADED")
        t = time()
        for i in xrange(iters):
            data_batch, label_batch = get_random_batch(data_subset,
                                                       chunk_size, batch_size)
            predictions = m.run(data_batch)
            if (i%100==0):
                print("reloaded model ran", i, "times")
                cm.add([INT2CLASS[x] for x in predictions], label_batch)
        print(cm)
        print("elapsed time:", time()-t)

def main():
    ### DATA LOADING
    DATA = get_dataset(DATASET_PATH, downsample_ratio=DOWNSAMPLE)
    TRAIN_SUBSET,CV_SUBSET,TEST_SUBSET = split_dataset(DATA,TRAIN_CV_TEST_RATIO,
                                                       CLASSES)
    del DATA # data won't be needed anymore and may free some useful RAM
    SAVE_PATH = os.path.join("./saved_models", make_timestamp())
    run_training(TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET, MODEL,
                 BATCH_SIZE, CHUNK_SIZE, MAX_STEPS,
                 L2_REG, OPTIMIZER_FN,
                 TRAIN_FREQ, CV_FREQ,
                 save_path=SAVE_PATH)
    ### LOAD AND USE TRAINED MODEL
    test_pretrained_model(TRAIN_SUBSET, SAVE_PATH,
                          CHUNK_SIZE, 1,50000)

def test_pretrained():
    """
    """
    run_pretrained_model(split_dataset(get_dataset(DATASET_PATH,
                                                   downsample_ratio=DOWNSAMPLE),
                                       TRAIN_CV_TEST_RATIO, CLASSES)[0],
                         "./saved_models/30_Nov_2017_10h23m58s", 6300, 1, 50000)

if __name__ == "__main__":
    # main()
    test_pretrained()
