# Simple Music Classifier

The Python + PyTorch approach.

## Prerequisites

- Using Linux or OS X (Sorry Windows users...)
- Python 2.7 installed
- Dataset GTZAN downloaded, unpacked and converted with au2wav.sh

## Setup

### Virtualenv

Place packages needed for this approach inside a virtualenv for easy cleanup / to avoid package conflicts.

Make sure your `python --version` is 2.7, you might need to use `python27` explicitly on some linux distros like Arch.

1. Install virtualenv through pip:

`pip install virtualenv`

2. Create a virtualenv

This will create a virtualenv called ".venv_torch" inside the current working directory.

`virtualenv -p python2 .venv_torch`

3. Activate virtualenv

Always do this before working on the project, because all dependencies will live inside the virtualenv.

`. .venv_torch/bin/activate`

4. How to deactivate virtualenv

To exit the virtualenv and restore the python environment to the previous, system-wide state, use:

`deactivate`

### Installing dependencies

Dependencies are listed inside `requirements.txt`. There are two versions, `requirements_CUDA.txt` installs PyTorch with CUDA 8.0 support. If you want CUDA support, but can only use 7.5 for some reason, use `requirements.txt`.

To install from requirements file:

`pip install -r requirements.txt`

## Project structure

The approach contains the following modules:

- `data.py` handles data preprocessing, random sampling
- `graph.py` contains the NN model
- `train.py` contains the logic for training the NN
- `main.py` runs the complete program

## Running the example

`python main.py`

Parameters can be tweaked in `main.py`