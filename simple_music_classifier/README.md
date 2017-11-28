### TODO:
  - [x] Find and download a suitable dataset: **GTZAN: 1.3GB, 1000 unique 30-sec chunks of 10 single-labeled genres**
  - [x] `tensorflow_approach_python`: Clean Andres' Python+TF code for a working minimal example that does preprocessing, GPU-training, evaluation and plotting
  - [x] `pytorch_approach`: Adapt TF code to work with PyTorch
  - [x] Explore further ML frameworks (Java client for TF, ???)
  - [x] Document installation process from clean virtual environments, and script them if possible. Also adapt to CPU?
  - [ ] Make slides (a latex template is already in the repo)


### BEFORE WE START:

0. This instructions are common to all approaches, and were tested on different linux distros. Other users install `cygwin`, virtual machine... and follow linux instructions (not tested).
1. clone this repositoriy
2. To get the GTZAN whole dataset, simply visit this link: `opihi.cs.uvic.ca/sound/genres.tar.gz`, and download it to `<THIS_REPO>/datasets/gtzan/`. Create the `datasets` folder if not existing:

```
<THIS_REPO>
          \
          |- README.md
          |+ simple_music_classifier
          |... (other folders and files)
          |+ datasets
                    \
                    | ... (other datasets...)
                    | + gtzan (download this folder from opihi.cs.uvic.ca/sound/genres.tar.gz)
                            \ 
                            |+ rock (folder with 100 .au files)
                            |+ reggae (folder with 100 .au files)
                            | ... (other genres)

```
3. Convert the audio files from .au to .wav. In each of the genre folders (rock, reggae...). In Linux this can be nicely done with [sox](https://ubuntuforums.org/showthread.php?t=1577560) `for i in *.au; do sox "$i" "${i%.au}.wav"; done && find . -type f -name "*.au" -delete` or alternatively run the `au2wav.sh` script that can be found in `<THIS_REPO>` **WARNING: the .au files will be deleted**.
4. The Python work will be performed on Python2.7. Make sure you have it installed, together with the `pip` package manager (ensure last pip is activated: `easy_install -U pip`, you may need `sudo` for this)
5. Also install virtualenv. 1 and 2 in ubuntu: `sudo apt-get install python-pip python-dev python-virtualenv`.
6. Create a virtual environment for the whole Python dependecies that we will need: `virtualenv -p python2 --system-site-packages ~/aioli_ve`
7. Before installing python packages and running python programs, log in to the venv with `source ~/aioli_ve/bin/activate`
8. First time you log in, make sure you have the last version of `pip` installed: `easy_install -U pip`
9. Make that you downloaded the GTZAN dataset to the `<THIS_REPO>/datasets/gztan` folder as explained before

Continue following the instructions to get started with the 

### THE `TENSORFLOW+PYTHON` APPROACH:

This approach is implemented in `<THIS_REPO>/simple_music_classifier/tensorflow_approach_python`, noted here as `<TFP_APPROACH>`

1. Log in the virtual environment as described before
2. Without GPU (CPU only): `pip install --upgrade tensorflow`. With GPU: see [this link](https://stackoverflow.com/a/47503155/4511978) for detailed instructions on how to install the latest CUDA+CUDNN+TensorFlow.
   1. If you want a TF version with the latest CPU instructions (may be a little faster) the installation process is a little more involved: `pip install https://github.com/mind/wheels/releases/download/tf1.4-cpu/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl`, and `sudo apt install mklibs` will probably do the trick.
3. Further requirements
```
pip install numpy
pip install scipy # just to load the .wav files as numpy array. A bit of overkill but convenient and "only" 50MB
pip install git+https://github.com/lanpa/tensorboard-pytorch # 
pip install tabulate
```
4. Test it!
```python
import tensorflow as tf
a = tf.constant(3)
b = tf.constant(4)
c = a+b
sess = tf.Session()
sess.run(c) # should output 7
```
5. Run the code! `cd <TF_APPROACH> && python main.py`

6. To visualize the training data with TensorBoard, run `tensorboard --logdir=<TFP_APPROACH>/runs` and visit `localhost:6006` with a valid web browser (google's chrome is usually well supported, others like firefox not so).
   1. To run tensorboard via SSH, see [here](https://stackoverflow.com/a/45024736)

### THE `PYTORCH` APPROACH:

1. Make sure you are logged in the virtual environment as described before.
2. The installation of pyorch is well documented in their web page [http://pytorch.org/](http://pytorch.org/)
