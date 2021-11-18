# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 15:23
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


import os
import re
import gc
import time
import glob
import h5py
import torch
import argparse
import warnings
import numpy as np
from torch import nn
from pssr import PSSR
from dataset import *
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Pytorch PSSR')
parser.add_argument('--model', default='PSSR', type=str, help='choose path of model')
parser.add_argument('--batchsize', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='number of train epoch')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--train_data', default='data/train/train.h5', type=str, help='path of train data')
parser.add_argument('--gpu', default='0,1', type=str, help='gpu id')
args = parser.parse_args()

cuda = torch.cuda.is_available()
nGPU = torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
savedir = os.path.join('model', args.model)


log_header = [
    'epoch',
    'iteration',
    'train/loss',
]

if not os.path.exists(savedir):
    os.mkdir(savedir)

if not os.path.exists(os.path.join(savedir, 'log.csv')):
            with open(os.path.join(savedir, 'log.csv'), 'w') as f:
                f.write(','.join(log_header) + '\n')


def find_checkpoint(savedir):
    file_list = glob.glob(os.path.join(savedir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for m in file_list:
            result = re.findall(".*model_(.*).pth.*", m)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def main():
    # load train data
    print('==> Data loading')
    hf = h5py.File(args.train_data, 'r+')
    lr_polsar = np.float32(hf['data'])
    hr_polsar = np.float32(hf['label'])
    lr_polsar = torch.from_numpy(lr_polsar).view(-1, 9, 20, 20)
    hr_polsar = torch.from_numpy(hr_polsar).view(-1, 9, 40, 40)
    train_set = PSSRDataset(lr_polsar, hr_polsar)
    train_loader = DataLoader(dataset=train_set, num_workers=8, drop_last=True, batch_size=64, shuffle=True, pin_memory=True)

    print('==> Model building')
    model = PSSR()
    criterion = nn.MSELoss(reduction='mean')
    if cuda:
        print('==> GPU setting')
        model = model.cuda()
        criterion = criterion.cuda()

    print('==> Optimizer setting')
    optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=0)

    initial_epoch = find_checkpoint(savedir=savedir)
    if initial_epoch > 0:
        print('==> Resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(savedir, 'model_%03d.pth' % initial_epoch))

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model)

    for epoch in range(initial_epoch, args.epoch):
        # scheduler.step(epoch)
        epoch_loss = 0
        start_time = time.time()

        # train
        model.train()
        for iteration, batch in enumerate(train_loader):
            lr_batch, hr_batch = Variable(batch[0]), Variable(batch[1])
            if cuda:
                lr_batch = lr_batch.cuda()
                hr_batch = hr_batch.cuda()
            optimizer.zero_grad()
            out = model(lr_batch)
            loss = criterion(out, hr_batch)
            epoch_loss += loss.data
            print('%4d %4d / %4d loss = %2.6f' % (epoch + 1, iteration, train_set.hr_train.size(0)/args.batchsize, loss.data))
            loss.backward()
            optimizer.step()
            with open(os.path.join(savedir, 'log.csv'), 'a') as file:
                log = [epoch+1, iteration] + [epoch_loss.data.item()]
                log = map(str, log)
                file.write(','.join(log) + '\n')

        if len(args.gpu) > 1:
            torch.save(model.module, os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        else:
            torch.save(model, os.path.join(savedir, 'model_%03d.pth' % (epoch + 1)))
        gc.collect()
        elapsed_time = time.time() - start_time
        print('epcoh = %4d , time is %4.4f s' % (epoch + 1, elapsed_time))


if __name__ == '__main__':
    main()
