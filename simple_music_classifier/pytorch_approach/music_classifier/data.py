import os
import random
from scipy.io import wavfile
import numpy as np
import torch
from torch.autograd import Variable
from tabulate import tabulate
import matplotlib.pyplot as plt


def random_wav_slice(path, slice_size, debug=False, category="bla", downsample=7):
    sample_rate, data = wavfile.read(path)
    data = data[::7]

    slice_start = random.randint(0, data.shape[0] - slice_size - 1)
    slice_end = slice_start + slice_size

    # save a snippet with cat label to listen to
    #wavfile.write("cat_%s.wav"%category, sample_rate, data[slice_start:slice_end])


    convert_16_bit = float(2**15)
    data = data[slice_start:slice_end]/(convert_16_bit+1.0)
    #data = (data[slice_start:slice_end] - np.min(data[slice_start:slice_end])) / (np.max(data[slice_start:slice_end]) - np.min(data[slice_start:slice_end]) + 1e-15)

    # save a plot of the data with category label
    '''
    fig = plt.figure(1)
    fig.clf()
    plt.title(category)
    plt.plot(data)
    plt.ylim((0,1))
    plt.savefig("cat_%s.png" % category)
    '''

    data = data.astype(np.float32)

    return data


def generate_categories(path):
    categories = dict.fromkeys(os.listdir(path))

    for c in categories:
        c_abspath = os.path.join(path, c)
        audio_files = filter(lambda n: n.endswith(".wav"), os.listdir(c_abspath))
        categories[c] = [os.path.join(c_abspath, au) for au in audio_files]

    return categories


def random_sample(categories, slice_size=20000, debug=False, batch_size=128):
    sample_tensors = []
    target_tensors = []
    sample_categories = []

    for batch_i in xrange(batch_size):

        category = random.choice(list(categories.keys()))
        category_index = sorted(list(categories.keys())).index(category)

        random_file = random.choice(list(categories[category]))
        sample_data = random_wav_slice(random_file, slice_size, category=category)

        target_tensors.append(category_index)
        sample_tensors.append(sample_data)
        sample_categories.append(category)

    sample_tensors = torch.from_numpy(np.array(sample_tensors, dtype=np.float32))
    target_tensors = torch.from_numpy(np.array(target_tensors, dtype=np.int64))

    return sample_categories, target_tensors, sample_tensors


def print_predictions(output_tensor, categories):
    values, indexes = output_tensor.topk(3, 1, True)

    for i in range(3):
        value = values[0][i].data[0]
        index = indexes[0][i].data[0]
        name = sorted(list(categories.keys()))[index]
        print("{}: {} ({})".format(i, name, value))

    print(80*"#")
    print("\n\n")


