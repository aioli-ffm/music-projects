### TODO:
  - [x] Find and download a suitable dataset: **GTZAN: 1.3GB, 1000 unique 30-sec chunks of 10 single-labeled genres**
  - [x] `tensorflow_approach_python`: Clean Andres' Python+TF code for a working minimal example that does preprocessing, GPU-training, evaluation and plotting
  - [x] `pytorch_approach`: Adapt TF code to work with PyTorch
  - [x] Explore Java client for TF
  - [x] Document installation process from clean virtual environments, and script them if possible. Also adapt to CPU?
  - [x] Make slides (a latex template is already in the repo)
  - [ ] Explore running TF models natively

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

Continue following the instructions to get started with the code examples.

### THE `PYTORCH` APPROACH:

1. Make sure you are logged in the virtual environment as described before.
2. The installation of pyorch is well documented in their web page [http://pytorch.org/](http://pytorch.org/)
3. To run the jupyter notebook, make sure that you have also sklearn installed: `pip install sklearn`

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

### THE `TENSORFLOW+JAVA` APPROACH:

The code in this section covers only loading and running a pre-trained TensorFlow model that was defined, pre-trained and saved with the TensorFlow+Python approach. This is convenient because the TF+Java API is far less developed and stable than the TF+Python API, and Java is usually wanted just for deploying and running the models.

1. Make sure you have the Java Development Toolkit 7 or higher installed. The installation details are well covered [here](https://docs.oracle.com/javase/8/docs/technotes/guides/install/install_overview.html).
2. You also need to install maven, which is also covered [here](https://maven.apache.org/install.html).
3. Run `cd <JAVA_APPROACH> && mvn install exec:java`. Done! You should see something like this:

```
Model evaluated 0 times
Model evaluated 1000 times
Model evaluated 2000 times
Model evaluated 3000 times
Model evaluated 4000 times
Model evaluated 5000 times
Model evaluated 6000 times
Model evaluated 7000 times
Model evaluated 8000 times
Model evaluated 9000 times
Model evaluated 10000 times
Model evaluated 11000 times
Model evaluated 12000 times
Model evaluated 13000 times
Model evaluated 14000 times
Model evaluated 15000 times
Model evaluated 16000 times
Model evaluated 17000 times
Model evaluated 18000 times
Model evaluated 19000 times
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time: 7.592 s
```


### THE `TENSORFLOW+NATIVE` APPROACH:

Since the backend for every TF model is C++, it should be possible to run it as a native application on several operative systems and embedded devices. TODO
