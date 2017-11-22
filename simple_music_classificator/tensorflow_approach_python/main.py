import scipy.io.wavfile as pywav
import numpy as np
import tensorflow as tf
from tensorboardX import FileWriter

import os
import random
from six.moves import xrange

################################################################################
# PREPROCESSING GLOBALS
################################################################################
DATASET_PATH = "../../datasets/gtzan/"
SAMPLE_RATE = 22050 # Nyquist = 11025Hz
TRAIN_CV_TEST_RATIO = [0.6, 0.1, 0.3]




################################################################################
# LOAD DATA: files are 22050Hz, 16-bit wavs
# 10 classes, 100 unique 30s wavs per class
# reggae, classical, country, jazz, metal, pop, disco, hiphop, rock, blues
################################################################################

def get_dataset(dataset_path):
    """Returns GTZAN dataset as a dictionary of key='classname',
       value='wavlist', where wavlist[XX] corresponds to
       'dataset_path/classname/classname.000XX.wav' as numpy array
    """
    data = {c:[] for c in os.listdir(dataset_path)}
    # fill the data dictionary: "DATASET_PATH/classname/classname.000XX.wav"
    for classname, wavlist in data.iteritems():
        class_path = os.path.join(dataset_path, classname)
        for i in xrange(100):
            wav_path = os.path.join(class_path, classname+"."+str(i).zfill(5))
            wavlist.append(pywav.read(wav_path+".wav")[1])
    return data

def downsample_nparray(arr, in_samplerate, out_samplerate):
    """Most of the style-defining features happen way below 10kHz frequency, so
       it could be interesting to train the classificator on downsampled the
       music.
    """
    NotImplemented



DATA = get_dataset(DATASET_PATH)
DATASET_CLASSES = data.keys()
# note that wav sizes differ slightly: range [660000, 675808] ~= (29.93s, 30.65)
wav_sizes = set([len(w) for wavlist in DATA.values() for w in wavlist])
MIN_CHUNKSIZE = min(wav_sizes)
MAX_CHUNKSIZE = max(wav_sizes)
print "wav sizes range (min, max): [%d, %d]\n" % (MIN_CHUNKSIZE, MAX_CHUNKSIZE)


################################################################################
# SPLIT DATA:
################################################################################

def split_dataset(dataset, train_cv_test_ratio):
    """Given a dictionary of key=class, value=wav_list, shuffles every wav_list
       and splits it by the proportions given by the [train, cv, test] ratio
       list (which must add up to 1), and returns the 3 subsets as 3 dicts in
       that order.
    """
    if sum(train_cv_test_ratio)!=1:
        raise RuntimeError("[ERROR] split_dataset: ratios don't add up to 1! ")
    train_subset = {}
    cv_subset = {}
    test_subset = {}
    for classname, wavlist in dataset.iteritems():
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


TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET = split_dataset(DATA, TRAIN_CV_TEST_RATIO)
del DATA # release unused reference just in case

# check that the proportions are correct
for c in TRAIN_SUBSET.iterkeys(): # c is classname
    print "  %s(train, cv, test): (%d, %d, %d)" % \
    (c, len(TRAIN_SUBSET[c]), len(CV_SUBSET[c]), len(TEST_SUBSET[c]))



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

weight_variable = lambda shape, stddev=0.1: tf.Variable(
    tf.truncated_normal(shape, stddev=stddev))
bias_variable = lambda shape: tf.Variable(tf.constant(0.1, shape=[shape]))

def basic_model(batch, num_classes, hidden_size=512):
    """A simple MLP. For every element of the input batch, performs:
       ========= LAYER 0 (input)==================================
                                        batch_size x chunk_size
       ========= LAYER 1 (hidden)=================================
       fully(chunk_sizexhidden_size)    batch_size x hidden_size
       ========= LAYER 2 (logits) ================================
       fully(hidden_size x num_classes) batch_size x num_classes
    """
    batch_size, chunk_size = batch.get_shape().as_list()
    #
    W1 = weight_variable([batch_size, hidden_size])
    b1 = bias_variable(hidden_size)
    out1 = relu(matmul(batch, W1)+b1)
    #
    W2 = weight_variable([hidden_size, num_classes])
    b2 = bias_variable(num_classes)
    logits = matmul(out1, W2) + b2
    return logits, l2loss(W1)+l2loss(W2)



################################################################################
# DEFINE TENSORFLOW GRAPH
################################################################################


def make_graph(model, chunk_shape, classes, l2reg=0):
    with tf.Graph().as_default():
        data_ph = tf.placeholder(tf.float16, shape=((None,)+chunk_shape),
                                 name="data_placeholder")
        labels_ph = tf.placeholder(tf.float16, shape=(None),
                                   name="labels_placeholder")
        logits, l2nodes = model(data_ph, len(classes), 512)
        loss = tf.reduce_mean(softmax(logits=data_ph, labels=labels_ph))
        if l2reg>0:
            loss += l2reg*l2nodes
        global_step = tf.Variable(0, name="global_step", trainable=False)
        minimizer = optimizer.minimize(loss, global_step=global_step)
        predictions = tf.argmax(logits, 1, output_type=tf.int32)
        #
        correct_predictions = tf.equal(predictions, labels_ph)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))






################################################################################
# DEFINE TF SESSION
################################################################################
CHUNK_SIZE = MIN_CHUNKSIZE
# reggae, classical, country, jazz, metal, pop, disco, hiphop, rock, blues
CLASSES = DATASET_CLASSES



if CHUNK_SIZE>MIN_CHUNKSIZE:
    raise RuntimeError("[ERROR] CHUNK_SIZE can't be bigger than MIN_CHUNKSIZE")


### MNIST
LOG_PATH = "../tensorboard_logs/"
TRAIN_MNIST_DDICT, CV_MNIST_DDICT, TEST_MNIST_DDICT = load_mnist_as_ddicts()
BATCHSIZE = 200
CHUNKSIZE = 28
# OPTMANAGER = OptimizerManager(tf.train.MomentumOptimizer, learning_rate=lambda x: 1e-4, momentum=lambda x:0.5)
OPTMANAGER = OptimizerManager(tf.train.AdamOptimizer, learning_rate=lambda x: 1e-4)
SELECTED_MODEL = mnist_model_conv
L2REG_FN = lambda x:  3e-07
DROPOUT_FN = lambda x: 1

with tf.Session()



def run_training(train_ddict, cv_ddict, test_ddict,
                 model, optimizer_manager,
                 batch_size, chunk_size, l2rate_fn, dropout_fn,
                 log_path,
                 batch_frequency=100, cv_frequency=500, snapshot_frequency=5000,
                 cv_voting_size=1, test_voting_size=1,
                 max_steps=float("inf"),
                 extra_info="",
                 normalize_chunks=True,
                 max_gpu_memory_fraction=1):
     # DATA HOUSEKEEPING: get num of classes, data shape, and create a bijection from its labels to
    # ascending ints starting by zero
    num_classes = len(train_ddict)
    data_shape = train_ddict.values()[0].values()[0].shape
    chunk_shape = (data_shape[0], chunk_size) if len(data_shape)==2 else (chunk_size,)
    class2int = {c:i for i, c in enumerate(train_ddict)}
    int2class = {v:k for k,v in class2int.iteritems()}
    # make graph
    g, graph_placeholders, graph_outputs = make_custom_graph(model, chunk_shape,
                                                             num_classes, optimizer_manager,
                                                             normalize_data=normalize_chunks)
    # run graph
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = max_gpu_memory_fraction
    with tf.Session(graph=g, config=sess_config) as sess:
        # session initialization
        sess.run(tf.global_variables_initializer())
        print("\nTF session initialized.\nMODEL:", model.__name__,
              "\nOPTIMIZER:", optimizer_manager,
              "\nBATCH SIZE:", batch_size,
              "\nCHUNK SHAPE:", chunk_shape,
              "\nL2 REGULARIZATION FACTOR:", l2rate_fn(0),
              "\nDROPOUT:", dropout_fn(0),
              "\nCV VOTING SIZE:", cv_voting_size,
              "\nTEST VOTING SIZE:", test_voting_size,
              "\nNORMALIZE CHUNKS", normalize_chunks)
        # tensorboard logging
        sess_datastring = make_session_datastring(model, optimizer_manager, batch_size, chunk_size,
                                                  l2rate_fn(0), dropout_fn(0), cv_voting_size,
                                                  test_voting_size, normalize_chunks, extra_info)
        log_dir = log_path + sess_datastring
        logger = TensorBoardLogger(log_path, sess_datastring, g)
        # start optimization
        step = 0
        while step<max_steps:
            step += 1
            # TRAINING
            train_data_batch, train_lbl_batch = get_random_batch(train_ddict, chunk_size, batch_size)
            train_lbl_batch = [class2int[lbl] for lbl in train_lbl_batch]
            train_feed = {graph_placeholders["data_placeholder"]:train_data_batch,
                          graph_placeholders["labels_placeholder"]:train_lbl_batch,
                          graph_placeholders["l2_rate"]: l2rate_fn(step),
                          graph_placeholders["dropout_rate"]:dropout_fn(step),
                          graph_placeholders["is_training"]:True}
            train_feed.update(optimizer_manager.feed_dict(step))
            _, lrate = sess.run([graph_outputs["minimizer"],
                                 graph_outputs["opt_lrate"]], feed_dict=train_feed)
            # plot training
            if step%batch_frequency == 0:
                plot_feed = {graph_placeholders["data_placeholder"]:train_data_batch,
                             graph_placeholders["labels_placeholder"]:train_lbl_batch,
                             graph_placeholders["l2_rate"]: l2rate_fn(step),
                             graph_placeholders["dropout_rate"]:1.0,
                             graph_placeholders["is_training"]:False}
                acc, lss = sess.run([graph_outputs["accuracy"],graph_outputs["loss"]], feed_dict=plot_feed)
                print("[TRAINING]","\tstep = "+str(step)+"/"+str(max_steps), "\tbatch_acc =", acc, "\tbatch_loss =", lss)
                logger.add_train_scalars(acc, lss, lrate, step)
            # CROSS VALIDATION
            if step%cv_frequency == 0:
                confmatrix = ConfusionMatrix(cv_ddict.keys(), "CV", voting_size=cv_voting_size)
                cv_total_loss = 0
                # split each CV sample into chunks and make a single batch with them:
                for ccc in cv_ddict:
                    for _, data in cv_ddict[ccc].iteritems():
                        cv_sample_data = cut_sample_to_chunks(data, chunk_size, shuffle=True)
                        cv_sample_len = len(cv_sample_data)
                        # pass the sample chunks to TF
                        for ii in xrange(0, cv_sample_len, cv_voting_size):
                            max_ii = min(ii+cv_voting_size, cv_sample_len) # avoids crash in last chunk
                            if(max_ii-ii<cv_voting_size):
                                print("CV warning: voting among", max_ii-ii, "elements, whereas",
                                      "voting size was", cv_voting_size,
                                      "(data_length, chunksize) =", (data.shape[1], chunk_size))
                            cv_data_batch = cv_sample_data[ii:max_ii]
                            cv_labels_batch = [class2int[ccc] for _ in xrange(len(cv_data_batch))]
                            cv_feed = {graph_placeholders["data_placeholder"]:cv_data_batch,
                                       graph_placeholders["labels_placeholder"]:cv_labels_batch,
                                       graph_placeholders["l2_rate"]: l2rate_fn(step),
                                       graph_placeholders["dropout_rate"]:1.0,
                                       graph_placeholders["is_training"]:False}
                            cv_preds, cv_lss = sess.run([graph_outputs["predictions"],
                                                         graph_outputs["loss"]], feed_dict=cv_feed)
                            cv_total_loss += cv_lss
                            confmatrix.add([int2class[vote(cv_preds)]], [ccc])
                print(confmatrix, "CV LOSS = ", cv_total_loss, sep="")
                logger.add_cv_scalars(confmatrix.accuracy()[0], cv_total_loss, step)
                logger.add_cv_confmatrix(confmatrix, step)
        # TESTING

        confmatrix = ConfusionMatrix(test_ddict.keys(), "TEST", voting_size=test_voting_size)
        test_total_loss = 0
        # split each TEST sample into chunks and make a single batch with them:
        for ccc in test_ddict:
            for _, data in test_ddict[ccc].iteritems():
                test_sample_data = cut_sample_to_chunks(data, chunk_size, shuffle=True)
                test_sample_len = len(test_sample_data)
                # pass the sample chunks to TF
                for ii in xrange(0, test_sample_len, test_voting_size):
                    max_ii = min(ii+test_voting_size, test_sample_len) # avoids crash in last chunk
                    if(max_ii-ii<test_voting_size):
                        print("TEST warning: voting among", max_ii-ii, "elements, whereas",
                              "voting size was", test_voting_size,
                              "(data_length, chunksize) =", (data.shape[1], chunk_size))
                    test_data_batch = test_sample_data[ii:max_ii]
                    test_labels_batch = [class2int[ccc] for _ in xrange(len(test_data_batch))]
                    test_feed = {graph_placeholders["data_placeholder"]:test_data_batch,
                               graph_placeholders["labels_placeholder"]:test_labels_batch,
                               graph_placeholders["l2_rate"]: l2rate_fn(step),
                               graph_placeholders["dropout_rate"]:1.0,
                               graph_placeholders["is_training"]:False}
                    test_preds, test_lss = sess.run([graph_outputs["predictions"],
                                                 graph_outputs["loss"]], feed_dict=test_feed)
                    test_total_loss += test_lss
                    confmatrix.add([int2class[vote(test_preds)]], [ccc])
        print(confmatrix, "TEST LOSS = ", test_total_loss, sep="")
        logger.add_test_scalars(confmatrix.accuracy()[0], test_total_loss, step)
        logger.add_test_confmatrix(confmatrix, step)
        logger.close()
