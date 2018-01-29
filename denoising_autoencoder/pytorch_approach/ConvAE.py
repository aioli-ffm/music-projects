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
from Datasets.datasets import MNIST
from PIL import Image
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, 8, stride=2, padding=1),
        #     nn.BatchNorm2d(64, eps=args.batch_norm),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 198, 8, stride=2, padding=1),
        #     nn.BatchNorm2d(198, eps=args.batch_norm),
        #     nn.ReLU(True),
        #     nn.Conv2d(198, 198, 5, stride=2)
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 198, 3, stride=1, padding=0),
            nn.BatchNorm2d(198),
            nn.ReLU(True),
            nn.Conv2d(198, 198, 5, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(198, 198, 5, stride=1),
            nn.BatchNorm2d(198),
            nn.ReLU(True),
            nn.ConvTranspose2d(198, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

best_prec1 = 0.
def train_val_net(data_path):
    global best_prec1
    best_prec1 = 0.
    model = CAE()
    # model = torch.nn.DataParallel(model)                                 ### Uncomment when cuda is used
    # model.cuda()
    # criterion = nn.BCELoss().cuda()
    criterion = nn.BCELoss()                                               ### Uncomment when cuda is used
    cudnn.benchmark = True
                 ### Yet to do the Xavier initialization ###

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001,\
                           betas = (0.9, 0.999), \
                           eps = 1e-08, weight_decay = 0.0005)

    traindir = os.path.join(data_path, 'train/')
    valdir = os.path.join(data_path, 'test/')

                        ### Don't Normalize
    # normalize = transforms.Normalize(mean=[0.46626906767843135, 0.3553785852011765, 0.353770305034902],    #mean
    #                                  std= [0.11063050470941177, 0.10724310918039215, 0.1115941032545098])  # and std for MNIST

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         # transforms.RandomHorizontalFlip(),                   # No mirroring
    #         transforms.ToTensor(),
    #         # normalize,                                           # No normalization
    #     ]))
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size = 128, shuffle = True, \
    #     num_workers = 4)
    #     # pin_memory = True, \
    #     # sampler = state_space_parameters.train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.ToTensor(),
    #         # normalize,                                           # No normalization
    #     ])),
    #     batch_size = 100, shuffle=False, \
    #     num_workers = 4)
    #     # pin_memory=True)

    writer = SummaryWriter()
    mnist = MNIST(False, 28, 128,\
                  100, 4, traindir, valdir)
    train_loader = mnist.train_loader
    val_loader = mnist.val_loader
    start_lr = 0.001
    train_flag = True
    epoch = 0
    restart = 0
    end_epoch = 20
    while epoch!=end_epoch:
        epoch += 1
        adjust_learning_rate(optimizer, epoch, start_lr)
        train(train_loader, model, criterion, optimizer, epoch)
        validate(val_loader, model, criterion, epoch == end_epoch, writer)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)                          ### Uncomment when cuda is used                       
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(input)
        output = model(input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

def validate(val_loader, model, criterion, print_flag, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)                              ### Uncomment when cuda is used
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(input, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        # save_image((input_var.data)[0].view(3, 28, 28), '/home/sagnik/Desktop/4_1_LOP/Thesis/RameshThesis/newCode/AutoEncoder/input.png')
        if i%10 == 0:
            x = vutils.make_grid(output.data)
            writer.add_image('Image', x, i)
        # save_image((output.data)[0].view(3, 28, 28), '/home/sagnik/Desktop/4_1_LOP/Thesis/RameshThesis/newCode/AutoEncoder/output.png')


    print(' validation_losses:{losses.avg:.3f}'
          .format(losses = losses))


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


def adjust_learning_rate(optimizer, epoch, training_lr):
    lr = training_lr * (0.2**(epoch//5))  # reduction by 0.2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train_val_net('/home/shared/sagnik/MNIST_full')