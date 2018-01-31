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
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image
import argparse
import numpy as np
import datetime # timestamp for sessions
from time import time # rough time measurements
import os # for listdir and path.join
import random
from six.moves import xrange
from tabulate import tabulate # pretty-print of confusion matrices on terminal
import scipy.io.wavfile as pywav
from scipy import signal # for downsampling
from tensorboardX import SummaryWriter
# another python file holding the model definitions
from libs.ConvAE import CAE

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
        audio_files = filter(lambda n: n.endswith(".wav"), os.listdir(class_path))      # added
        # for i in xrange(100):                                                         # commented
        for i in xrange(len(audio_files)):                                              # added
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
# AVERAGE METER
################################################################################


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
################################################################################
# TRAINING PyT SESSION
################################################################################


def make_timestamp2():
    """Sample output: 01_Jan_2019_12h00m05s
    """
    return '{:%d_%b_%Y_%Hh%Mm%Ss}'.format(datetime.datetime.now())

''' optimizer needs to be changed '''
def run_training(train_subset, cv_subset, test_subset, model,
                 batch_size, chunk_size, make_timestamp, max_steps=float("inf"),
                 l2reg=0, optimizer_fn='optim.Adam',                                      # changed
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
    logger = SummaryWriter('/home/shared/sagnik/output/tensorboard/AE_MusicDenoising/' + \
                            make_timestamp + '_GTZAN_ConvAE')
    
    logger.add_text("Session info", make_timestamp +
                    " model=%s batchsz=%d chunksz=%d l2reg=%f" %
                    (model.__name__, batch_size, chunk_size, l2reg), 0)
    model = CAE()
    model = torch.nn.DataParallel(model)                                 ### Uncomment when cuda is used
    model.cuda()
    # criterion = nn.BCELoss()
    criterion = nn.BCELoss().cuda()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001,\
                               betas = (0.9, 0.999), \
                               eps = 1e-08, weight_decay = l2reg)
    cudnn.benchmark = True

    train_losses = AverageMeter()
    cv_losses_total = AverageMeter()
    for step in range(max_steps):
      data_batch, label_batch = get_random_batch(train_subset, chunk_size,
                                                 batch_size)
      data_batch = data_batch.reshape(data_batch.shape[0],1,-1,1)
      data_batch_torch = torch.from_numpy(data_batch)
      lbl = [CLASS2INT[l] for l in label_batch] # convert from str to int

      # input_var = torch.autograd.Variable(data_batch_torch)
      input_var = torch.autograd.Variable(data_batch_torch).cuda()
      # target_var = torch.autograd.Variable(data_batch_torch)
      target_var = torch.autograd.Variable(data_batch_torch).cuda()
      output = model(input_var)
      # if step == 0:
      loss = criterion(output, target_var)
      # else:
      #   loss.data = loss.data + criterion(output, target_var).data
      train_losses.update(criterion(output, target_var).data[0], data_batch_torch.size(0))

      if(step%train_freq==0):
        model.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      if((step-1)%train_freq == 0):
        pass

      if(step%cv_freq==0):
        model.eval()
        cv_losses = AverageMeter()
        for c_cv in train_subset:
          # print("validating class %s: this may take a while..."%c_cv)
          # extract ALL chunks of selected class and corresponding lbl
          cv_data = get_class_batch(cv_subset, c_cv, chunk_size)
          cv_data = cv_data.reshape(-1,1, 6300,1)
          cv_data_torch = torch.from_numpy(cv_data)
          cv_labels = cv_data_torch
          # input_var = Variable(cv_data_torch)
          input_var = Variable(cv_data_torch, volatile = True).cuda()
          # target_var = Variable(cv_labels)
          target_var = Variable(cv_labels, volatile = True).cuda()
          output = model(input_var)
          loss = criterion(output, target_var)
          cv_losses.update(loss.data[0], cv_data_torch.size(0))
          cv_losses_total.update(loss.data[0], cv_data_torch.size(0))
      if((step - 1)%cv_freq == 0):
        print('Average total validation losses after {} steps of validation:{}'.format(step, cv_losses_total.avg))

    print('final average training loss:{} and final average validation loss:{}'.\
          format(train_losses.avg, cv_losses_total.avg))



################################################################################
# RUN ALL
################################################################################

# GLOBALS AND HYPERPARAMETERS

''' needs to be changed '''
DATASET_PATH = "/home/shared/sagnik/datasets/gtzan/"                                              

TRAIN_CV_TEST_RATIO = [0.7, 0.1, 0.2]                                               
CLASSES =  ["reggae", "classical", "country", "jazz", "metal", "pop", "disco",
            "hiphop", "rock", "blues"]
CLASS2INT = {"reggae":0, "classical":1, "country":2, "jazz":3, "metal":4,
             "pop":5, "disco":6, "hiphop":7, "rock":8, "blues":9}
INT2CLASS = {v:k for k,v in CLASS2INT.iteritems()}


MODEL=  CAE                                                           

DOWNSAMPLE= 7
BATCH_SIZE = 100
CHUNK_SIZE = (22050*2)//DOWNSAMPLE
MAX_STEPS=10001
L2_REG = 1e-3


OPTIMIZER_FN = 'optim.Adam'                                                          

TRAIN_FREQ=1
CV_FREQ=100
parser = argparse.ArgumentParser(description='Music Reconstruction and Denoising')
parser.add_argument('data', metavar='DIR', default='/home/shared/sagnik/datasets/gtzan' , 
                    help='path to dataset')
parser.add_argument('max_steps', metavar = 'MAX_STEPS', default = 1001, help='maximum no. of iterations')
parser.add_argument('batch_size', metavar = 'BATCH_SIZE', default = 1000, help = 'batch size for training')
parser.add_argument('cv_freq', metavar = 'CV_FREQ', default = 100, help = 'frequency of validation')

def main():
    ### DATA LOADING
    global args
    args = parser.parse_args()
    DATASET_PATH = args.data
    MAX_STEPS = int(args.max_steps)
    BATCH_SIZE = int(args.batch_size)
    CV_FREQ = int(args.cv_freq)
    DATA = get_dataset(DATASET_PATH, downsample_ratio=DOWNSAMPLE)
    TRAIN_SUBSET,CV_SUBSET,TEST_SUBSET = split_dataset(DATA,TRAIN_CV_TEST_RATIO,
                                                       CLASSES)
    del DATA # data won't be needed anymore and may free some useful RAM
    make_timestamp = make_timestamp2()

    ''' needs to be changed '''
    SAVE_PATH = os.path.join("/home/shared/sagnik/output/saved_models", make_timestamp)                    # changed

    run_training(TRAIN_SUBSET, CV_SUBSET, TEST_SUBSET, MODEL,
                 BATCH_SIZE, CHUNK_SIZE, make_timestamp, MAX_STEPS,
                 L2_REG, OPTIMIZER_FN,
                 TRAIN_FREQ, CV_FREQ,
                 save_path=SAVE_PATH)
    ### LOAD AND USE TRAINED MODEL
    test_pretrained_model(TRAIN_SUBSET, SAVE_PATH,
                          CHUNK_SIZE, 1,50000)


main()
