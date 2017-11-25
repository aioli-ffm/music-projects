### TODO:
  - [x] Find and download a suitable dataset: **GTZAN: 1.3GB, 1000 unique 30-sec chunks of 10 single-labeled genres**
  - [x] `tensorflow_approach_python`: Clean Andres' Python+TF code for a working minimal example that does preprocessing, GPU-training, evaluation and plotting
  - [ ] `pytorch_approach`: Adapt TF code to work with PyTorch
  - [ ] Explore further ML frameworks (Java client for TF, ???)
  - [ ] Document installation process from clean virtual environments, and script them if possible. Also adapt to CPU?
  - [ ] Make slides (a latex template is already in the repo)


### BEFORE WE START:

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
3. Convert the audio files from .au to .wav. In each of the genre folders (rock, reggae...): `for i in *.au; do sox "$i" "${i%.au}.wav"; done && find . -type f -name "*.au" -delete`
4. The Python work will be performed on Python2.7. Make sure you have it installed, together with the `pip` package manager (ensure last pip is activated: `easy_install -U pip`)
5. Also install virtualenv. 1 and 2 in ubuntu: `sudo apt-get install python-pip python-dev python-virtualenv`.
6. Create a virtual environment for the whole Python dependecies that we will need: `virtualenv --system-site-packages ~/aioli_ve`
7. Before installing python packages and running python programs, log in to the venv with `source ~/aioli_ve/bin/activate`
8. First time you log in, make sure you have the last version of `pip` installed: `easy_install -U pip`
9. Make that you downloaded the GTZAN dataset to the `<THIS_REPO>/datasets/gztan` folder as explained before


### THE `TENSORFLOW+PYTHON` APPROACH:

1. Log in the virtual environment as described before
2. With GPU: `pip install --upgrade tensorflow-gpu`. Without GPU (CPU only): `pip install --upgrade tensorflow`
   1. TensorFlow+GPU needs the CUDA+CUDNN libraries. See [this link](https://www.tensorflow.org/install/install_linux#nvidia_requirements_to_run_tensorflow_with_gpu_support).
3. Further requirements
```
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
sess.run(c)
```
5. Run the code! `python <THIS_REPO>/simple_music_classifier/tensorflow_approach_python/main.py`

### THE `PYTORCH` APPROACH:

1. asereje
2. ja 
3. deje


## TODO SAMSTAG:
1. remove scipy dependencies and au->wav conversion
