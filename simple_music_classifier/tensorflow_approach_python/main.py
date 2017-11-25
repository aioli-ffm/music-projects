from __future__ import print_function, division
import scipy.io.wavfile as pywav
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from tabulate import tabulate

import os
import random
from six.moves import xrange
from scipy import signal




################################################################################
# PREPROCESSING GLOBALS
################################################################################

DATASET_PATH = "../../datasets/gtzan/"
SAMPLE_RATE = 22050
TRAIN_CV_TEST_RATIO = [0.7, 0.1, 0.2]





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
            # arr = pywav.read(wav_path+".wav")[1].astype(np.float32)
            # if normalize: arr /= max(abs(arr.max()), abs(arr.min()))
            arr = np.array([-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0], dtype=np.float32)
            print(">>>>>>>>>> BEFORE:", arr[0:20], arr.dtype)
            if downsample > 0:
                arr = signal.decimate(arr, downsample)
                print(">>>>>>>>>> AFTER:", arr[0:20], arr.dtype)
                raw_input("yo mama is so fat")
            wavlist.append(arr)
    return data

# DATA = get_dataset(DATASET_PATH)
# DATASET_CLASSES = DATA.keys()
# # note that wav sizes differ slightly: range [660000, 675808]~=(29.93s, 30.65)
# wav_sizes = set([len(w) for wavlist in DATA.values() for w in wavlist])
# MIN_CHUNKSIZE = min(wav_sizes)
# MAX_CHUNKSIZE = max(wav_sizes)
# print("wav sizes range (min, max): [%d, %d]\n"%(MIN_CHUNKSIZE, MAX_CHUNKSIZE))





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

# TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET = split_dataset(DATA, TRAIN_CV_TEST_RATIO, ["reggae", "classical"])
# del DATA # release unused reference just in case

# # check that the proportions are correct
# for c in TRAIN_SUBSET.iterkeys(): # c is classname
#     print("  %s(train, cv, test): (%d, %d, %d)" %
#           (c, len(TRAIN_SUBSET[c]), len(CV_SUBSET[c]), len(TEST_SUBSET[c])))





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
    return data, labels # cast data to float32?

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

# aliases
matmul = tf.matmul
relu = tf.nn.relu
dropout = tf.nn.dropout
l2loss = tf.nn.l2_loss
softmax = tf.nn.sparse_softmax_cross_entropy_with_logits
# conv1d = tf.nn.conv1d
# conv2d = tf.nn.conv2d
# max_pool = tf.nn.max_pool
# batch_norm = tf.layers.batch_normalization

def weight_variable(shape, stddev=0.000, dtype=tf.float32):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=dtype))

def bias_variable(shape, dtype=tf.float32):
    return tf.Variable(tf.constant(0, shape=[shape], dtype=dtype))

def simple_mlp(batch, num_classes, hidden_size=64):
    """A simple MLP. For every element of the input batch, performs:
       ========= LAYER 0 (input)==================================
                                         batch_size x chunk_size
       ========= LAYER 1 (hidden)=================================
       fully(chunk_sizexhidden_size)     batch_size x hidden_size
       ========= LAYER 2 (logits) ================================
       fully(hidden_size x num_classes ) batch_size x num_classes
    """
    # add one extra dim at the end (needed by matmul)
    batch_size, chunk_size = batch.get_shape().as_list()
    #
    W1 = weight_variable([chunk_size, hidden_size], dtype=tf.float16)
    b1 = bias_variable(hidden_size, dtype=tf.float16)
    out1 = relu(matmul(batch, W1)+b1)
    #
    W2 = weight_variable([hidden_size, num_classes], dtype=tf.float16)
    b2 = bias_variable(num_classes, dtype=tf.float16)
    logits = matmul(out1, W2) + b2
    return logits, l2loss(W1)+l2loss(W2)




################################################################################
# DEFINE TENSORFLOW GRAPH
################################################################################


def make_graph(model, chunk_shape, num_classes, l2reg=0,
               optimizer_fn=lambda:tf.train.AdamOptimizer()):
    with tf.Graph().as_default() as g:
        data_ph = tf.placeholder(tf.float16, shape=((None,)+chunk_shape),
                                 name="data_placeholder")
        labels_ph = tf.placeholder(tf.int32, shape=(None),
                                   name="labels_placeholder")
        logits, l2nodes = model(data_ph, num_classes)
        loss = tf.reduce_mean(softmax(logits=logits, labels=labels_ph))
        if l2reg>0:
            loss += l2reg*l2nodes
        global_step = tf.Variable(0, name="global_step", trainable=False)
        minimizer = optimizer_fn().minimize(loss, global_step=global_step)
        predictions = tf.argmax(logits, 1, output_type=tf.int32)
        return g, [data_ph, labels_ph], [logits, loss,global_step,minimizer,
                                         predictions]




################################################################################
# DEFINE TF SESSION
################################################################################

def run_training(train_subset, cv_subset, test_subset, model,
                 batch_size, chunk_size, max_steps=float("inf"),
                 l2reg=0, optimizer_fn=lambda: tf.train.AdamOptimizer(),
                 train_freq=10, cv_freq=100):
    # get current classes and map them to 0,1,2,3... integers
    classes = {c:i for i, c in enumerate(train_subset.keys())}
    classes_inv = {v:k for k,v in classes.iteritems()}
    # MATRIX OBJECTS (for the metrics) AND LOGGER (to plot them to TENSORBOARD)
    logger = SummaryWriter()
    # CREATE TF GRAPH
    graph,[data_ph,labels_ph],[logits,loss,global_step,minimizer,predictions]=(
        make_graph(model, (chunk_size,), len(CLASSES), l2reg, optimizer_fn))
    # START SESSION (log_device_placement=True)
    with tf.Session(graph=graph, config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer()) # compiler will warn you
        # START TRAINING
        for step in xrange(max_steps):
            data_batch, label_batch = get_random_batch(train_subset, chunk_size,
                                                       batch_size)
            lbl = [classes[l] for l in label_batch] # convert from str to int
            sess.run(minimizer, feed_dict={data_ph:data_batch, labels_ph:lbl})
            # LOG TRAINING
            if(step%train_freq==0):
                p, l, i, logts = sess.run([predictions,loss,global_step,logits],
                                feed_dict={data_ph:data_batch, labels_ph:lbl})
                cm_train = ConfusionMatrix(classes.keys(), "BATCH")
                cm_train.add([classes_inv[x] for x in p], label_batch)
                acc, by_class_acc = cm_train.accuracy()
                print(cm_train)
                print(">>>>>>", logts)
                logger.add_scalars("TRAINING", {"acc":acc, "loss":l}, i)
            # LOG CROSS-VALIDATION
            if(step%cv_freq==0):
                cv_total_loss = 0
                cm_cv = ConfusionMatrix(classes.keys(), "CV")
                for c_cv in classes:
                    print("validating class %s: this may take a while..."%c_cv)
                    # extract ALL chunks of selected class and corresponding lbl
                    cv_data = get_class_batch(cv_subset, c_cv, chunk_size)
                    cv_labels = [c_cv for _ in xrange(len(cv_data))]
                    lbl = [classes[c_cv] for _ in xrange(len(cv_data))]
                    p,l,i = sess.run([predictions, loss, global_step],
                                     feed_dict={data_ph:cv_data, labels_ph:lbl})
                    cv_total_loss += l
                    cm_cv.add([classes_inv[x] for x in p], cv_labels)
                # once loss and matrix has been calculated for every class...
                acc, by_class_acc = cm_cv.accuracy()
                logger.add_scalars("CV", {"acc":acc, "loss":cv_total_loss}, i)
        # AFTER TRAINING LOOP ENDS, DO VALIDATION ON THE TEST SUBSET (omitted
        # here for brevity, code is identical to the one for cross-validation)
        print("here could be your amazing test with 99.9% accuracy!!")
        return





################################################################################
# RUN TF SESSION
################################################################################


# SET HYPERPARAMETERS ##########################################################
# reggae, classical, country, jazz, metal, pop, disco, hiphop, rock, blues
CLASSES = ["reggae", "classical"]
MODEL= lambda batch, num_classes: simple_mlp(batch, num_classes, 1024)
BATCH_SIZE = 50
CHUNK_SIZE = 22050 # for GTZAN(22050) this has to be smaller than 660000
MAX_STEPS=10000
L2_REG = 0
OPTIMIZER_FN = lambda: tf.train.AdamOptimizer(1e-5)
TRAIN_FREQ=5
CV_FREQ=50
DOWNSAMPLE=2
################################################################################
DATA = get_dataset(DATASET_PATH, downsample=DOWNSAMPLE)
print(">>>>>>>>>>", DATA.values()[0][0].dtype)
raw_input("stop")
TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET = split_dataset(DATA, TRAIN_CV_TEST_RATIO,
                                                     CLASSES)

# CV_SUBSET = {k:v[0:2] for k,v in CV_SUBSET.iteritems()}
del DATA # data won't be needed anymore and may free some useful RAM

# check that the proportions are correct
for c in TRAIN_SUBSET.iterkeys(): # c is classname
    print("  %s(train, cv, test): (%d, %d, %d)" %
          (c, len(TRAIN_SUBSET[c]), len(CV_SUBSET[c]), len(TEST_SUBSET[c])))


run_training(TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET, MODEL,
             BATCH_SIZE, CHUNK_SIZE, MAX_STEPS,
             L2_REG, OPTIMIZER_FN,
             TRAIN_FREQ, CV_FREQ)
