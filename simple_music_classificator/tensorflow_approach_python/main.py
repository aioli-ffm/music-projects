import scipy.io.wavfile as pywav
import tensorflow as tf
from tensorboardX import FileWriter
import os


################################################################################
# PREPROCESSING GLOBALS
################################################################################
DATASET_PATH = "../../datasets/gtzan/"
SAMPLE_RATE = 22050 # Hz
TRAIN_CV_TEST_RATIO = [0.6, 0.1, 0.3]




################################################################################
# LOAD DATA: files are 22050Hz, 16-bit wavs
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
        for i in range(100):
            wav_path = os.path.join(class_path, classname+"."+str(i).zfill(5))
            wavlist.append(pywav.read(wav_path+".wav")[1])

DATA = get_dataset(DATASET_PATH)

# note that wav sizes differ slightly: range [660000, 675808] ~= (29.93s, 30.65)
wav_sizes = set([len(w) for wavlist in DATA.values() for w in wavlist])
print "wav sizes range: [%d, %d]"% (min(wav_sizes), max(wav_sizes))




################################################################################
# SPLIT DATA: 10 classes, 100 unique 30s wavs per class
################################################################################

def split_dataset(train_cv_test_ratio):
    if sum(train_cv_test_ratio)!=1:
        raise RuntimeError("[ERROR] split_dataset: ratios don't add up to 1! "
