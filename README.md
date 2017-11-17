
# AIOLI-MUSIC:

This README file contains information about the music-related resources and work done in the AIOLI.

## TODO

#### THIS IS GOING TO HAPPEN
- [ ] Initial presentation on a **SIMPLE CLASSIFIER USING DIFFERENT ML FRAMEWORKS**, 29th November 2017 (Joshi, Kyle, Noah, Andres, ...)
- [ ] ...?
- [ ] Profit!


#### THIS COULD HAPPEN AND WOULD BE AWESOME
* Music denoising using autoencoders `https://github.com/spmallick/learnopencv/blob/master/DenoisingAutoencoder/Denoising-Autoencoder-using-Tensorflow.ipynb`















* Source separation using autoencoders
* Emulate Spotify's recommender system using CNNs: `https://hackernoon.com/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe`
* Music style transfer (decomposition using wavelets, recomposition using SuperCollider: `http://supercollider.github.io/`, mapping using some unsupervised/clustering algorithms)

## 1. DATASETS

This is a selection of the most popular datasets. Most of them contain "songs" as understood in the context of 
For a single-label, small setup to train some music classification supervised algos, both the GTZAN dataset and the FMA-small version are suitable. The FMA has less research on it (AFAIK), but the GTZAN lacks licensing.

For a big-scale, multi-label project, the MSD is the absolute reference but also receives critique for not holding the plain audio files. The magnatagatune dataset has some research done on it and seems a good choice.

Some may have a normalized dabase system avaliable and precomputed features, but I think it is especially interesting to get the raw musical data and get through step of the pipeline from there.

#### FREE MUSIC ARCHIVE:

* CC-inspired license
* four versions (from 8K 30sec tracks on 8 balanced genres to 100K full tracks on 161 unbalanced genres)
* audio as well as higher-level features
* repo with supporting code in Python2

paper: `https://arxiv.org/abs/1612.01840`
code and downloads: `https://github.com/mdeff/fma` 
more info: `https://freemusicarchive.org/api`


#### GTZAN DATASET:

* 1000 audio tracks of 30 seconds each
* 10 genres, each 100 tracks
* 22050Hz mono, 16-bit .wav files
* presumably no license given (but not needed for 30sec?)
* much research done on it
webpage and downloads: `http://marsyasweb.appspot.com/download/data_sets/`

Note: the web has also a speech vs. music dataset avaliable

#### MAGNATAGATUNE DATASET:

* over 25k 30s chunks of mp3 files
* multi-label features over 220 categories (water, english, upbeat, quick...)

webpage and downloads: `http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset` 

#### MILLION SONG DATASET:

* mid- and high-level features of a million whole songs
* no audio, but open license (many of them are "commercially relevant")
* lots of research done on it

#### CALAB DATASET:

* over 10,000 songs performed by 4,597 different artists, weakly labeled from a vocabulary of over 500 tags
* song-tag associations are mined from Pandora's website
* specific format and length for the contents +licensing unclear, read `http://calab1.ucsd.edu/~datasets/cal500/details_cal500.txt`

downloads: `http://calab1.ucsd.edu/~datasets/`









## 2. AUDIO PREPROCESSING IN PYTHON 2:

* NumPy is the omnipresent python library for numerical computation. It features tensors of arbitrary rank and a broad set of efficiently implemented operations on them, as well as vectorized notation. Most python-based machine learning frameworks (like TensorFlow or PyTorch) interact closely with it.

* Importing and exporting wave files: `import scipy.io.wavfile as pywav`
* Loading .au files: `https://help.scilab.org/doc/6.0.0/en_US/auread.html`

* LibRosa is a library for extracting audio features (especially time/frequency representations): `https://github.com/librosa/librosa` 

This libraries are mostly oriented to CPU work. It is also possible to perform preprocessing and data augmentation on the GPU, but for most musical applications these are fine.

See the related files in this repo for more details.





## 3. TENSORFLOW, PYTORCH, TENSORBOARD: INSTALLATION (VENV) AND USAGE

Not sure if this is the place to refer to this since it is not specific to music. Check the related files in the repo for the details!





