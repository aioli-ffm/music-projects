import os
import random
from scipy.io import wavfile
import numpy as np
import torch
from torch.autograd import Variable
from tabulate import tabulate


def random_wav_slice(path, slice_size, debug=False):
    """
    :param path: str Filesystem path to a .wav file
    :return: np array of dimension #channels x slice_size
    """
    sample_rate, data = wavfile.read(path)

    slice_start = random.randint(0, data.shape[0] - slice_size - 1)
    slice_end = slice_start + slice_size

    data = data[slice_start:slice_end]

    # Normalize loudness, very dirty
    """peak = max(data.max(), abs(data.min()))
    if peak == 0:
        peak = 1
    data = data.astype(np.float16) / peak"""

    # force dimension
    data = np.array([data]).astype(np.float32)

    return data


def generate_categories(path):
    """
    Generates a dict of categories in our dataset.
    Each dict key contains a list of paths where the sample files
    can be found.
    :param path: str Path to the dataset
    """
    categories = dict.fromkeys(os.listdir(path))

    for c in categories:
        c_abspath = os.path.join(path, c)
        audio_files = filter(lambda n: n.endswith(".wav"), os.listdir(c_abspath))
        categories[c] = [os.path.join(c_abspath, au) for au in audio_files]

    return categories


def random_sample(categories, slice_size=20000, debug=False):
    """
    Given a category dict from generate_categories, this will choose
    a random sample file, grab a random slice of the sample file and
    package it up into a torch.autograd.Variable containing a torch.Tensor.
    The Dimensions of the resulting Tensor will be:
    batch_size x nchannels x slice_size
    Because we don't use batches and our sources are mono, this simplifies to:
    1 x 1 x slice_size
    :param categories: dict Of categories
    :param slice_size: int Number of values in sample
    """
    category = random.choice(list(categories.keys()))
    category_index = sorted(list(categories.keys())).index(category)
    #category_tensor = torch.zeros(len(categories))
    #category_tensor[category_index] = 1.0
    category_tensor = torch.LongTensor([category_index])

    random_file = random.choice(list(categories[category]))
    # force extra dimension for batch
    sample_data = np.array([random_wav_slice(random_file, slice_size)])
    sample_tensor = torch.from_numpy(sample_data)

    return (category_index, category), Variable(category_tensor), \
    Variable(sample_tensor)


def print_predictions(output_tensor, categories):
    values, indexes = output_tensor.topk(3, 1, True)

    for i in range(3):
        value = values[0][i].data[0]
        index = indexes[0][i].data[0]
        name = sorted(list(categories.keys()))[index]
        print("{}: {} ({})".format(i, name, value))

    print(80*"#")
    print("\n\n")


class ConfMatrix():
    def __init__(self, categories):
        self.cats = sorted(list(categories.keys()))
        self.buckets = dict.fromkeys(categories.keys())
        for cat in self.buckets.iterkeys():
            self.buckets[cat] = dict.fromkeys(categories.keys())
            for c in self.buckets[cat].iterkeys():
                self.buckets[cat][c] = 0
        self.maxes = dict.fromkeys(categories.keys(), 0)

    def category_name_from_tensor(self, output_tensor):
        _, indexes = output_tensor.topk(1, 1, True)
        index = indexes[0][0].data[0]
        return self.cats[index]

    def add_prediction(self, output_tensor, target_category):
        predicted_name = self.category_name_from_tensor(output_tensor)
        self.buckets[target_category][predicted_name] += 1
        self.maxes[target_category] += 1

    def __repr__(self):

        header = sorted(list(self.buckets.keys()))
        table = []

        for cat in header:
            row = [cat]
            for cat2 in header:
                row.append(float(self.buckets[cat][cat2]) / self.maxes[cat])
            table.append(row)

        return tabulate(table, headers=["in/pred"] + header)