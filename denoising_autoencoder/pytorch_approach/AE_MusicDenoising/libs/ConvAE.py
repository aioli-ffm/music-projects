import argparse
import os
import shutil
import time
import collections
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from time import gmtime, strftime

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        conv_no = pool_no = fc_no = relu_no = drop_no = 0
        encoder_list = decoder_list = list()
        encoder_list.append(('conv1',nn.Conv2d(1, 32, (3,1), stride=1, padding= ((3-1)/2, (1-1)/2))))      # padding = (kernel_size-1)/2
        encoder_list.append(('leaky_relu1', nn.LeakyReLU(0.1)))
        encoder_list.append(('pool1', nn.MaxPool2d((2,1), (2,1))))
        encoder_list.append(('conv2',nn.Conv2d(32, 32, (3,1), stride=1, padding= ((3-1)/2, (1-1)/2))))      # padding = (kernel_size-1)/2
        encoder_list.append(('leaky_relu2', nn.LeakyReLU(0.1)))
        encoder_list.append(('encoding', nn.MaxPool2d((2,1), (2,1))))

        decoder_list.append(('conv3', nn.Conv2d(32, 32, (3,1), stride=1, padding= ((3-1)/2, (1-1)/2))))
        decoder_list.append(('leaky_relu3', nn.LeakyReLU(0.1)))
        decoder_list.append(('upsample1', nn.ConvTranspose2d(32, 32, (3,1), stride=(2,1), padding = ((3-1)/2, (1-1)/2), output_padding = (1,0))))       # making it symmetric
        decoder_list.append(('upsample2', nn.ConvTranspose2d(32, 32, (3,1), stride=(2,1), padding = ((3-1)/2, (1-1)/2), output_padding = (1,0))))
        decoder_list.append(('logits', nn.Conv2d(32, 1, (3,1), stride=1, padding= ((3-1)/2, (1-1)/2))))    # no_output_channels = 3
        decoder_list.append(('recon', nn.Sigmoid()))
        ''' Old auto-encoder: works fine on MNIST reconstruction '''
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 32, 3, stride=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 198, 3, stride=1, padding=0),
        #     nn.BatchNorm2d(198),
        #     nn.ReLU(True),
        #     nn.Conv2d(198, 198, 5, stride=1)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(198, 198, 5, stride=1),
        #     nn.BatchNorm2d(198),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(198, 64, 3, stride=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 3, 3, stride=1, padding=0),
        #     nn.Sigmoid()
        # )
        self.encoder = nn.Sequential(collections.OrderedDict(encoder_list))
        self.decoder = nn.Sequential(collections.OrderedDict(decoder_list))
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
