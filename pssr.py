# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 15:23
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


import torch
from torch import nn
import torch.nn.init as init


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.conv(x))


class ComplexBlock(nn.Module):
    def __init__(self):
        super(ComplexBlock, self).__init__()
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())
        self.convs3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())
        self.convs4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())
        self.convs5 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())
        self.convs6 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())
        self.conv_cb = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())

    def forward(self, x):
        conv1 = self.convs1(torch.index_select(x, 1, torch.LongTensor([0]).cuda()))
        conv2 = self.convs2(torch.index_select(x, 1, torch.LongTensor([1, 2]).cuda()))
        conv3 = self.convs3(torch.index_select(x, 1, torch.LongTensor([3, 4]).cuda()))
        conv4 = self.convs4(torch.index_select(x, 1, torch.LongTensor([5]).cuda()))
        conv5 = self.convs5(torch.index_select(x, 1, torch.LongTensor([6, 7]).cuda()))
        conv6 = self.convs6(torch.index_select(x, 1, torch.LongTensor([8]).cuda()))
        out = self.conv_cb(torch.cat((conv1, conv2, conv3, conv4, conv5, conv6), 1))

        return out


class PSSR(nn.Module):
    def __init__(self):
        super(PSSR, self).__init__()
        self.pssr_deconv = torch.nn.ConvTranspose2d(in_channels=9, out_channels=9, kernel_size=4, stride=2, padding=1, bias=False)
        self.pssr_cb = ComplexBlock()
        self.pssr_convb = self.make_layer(ConvBlock, 19)
        self.pssr_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())
        self.pssr_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU())
        self.initialize_weights()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.pssr_deconv(x)
        out2 = self.pssr_convb(self.pssr_cb(out1))
        out3 = self.pssr_conv1(out2)
        out4 = torch.add(out1, out3)
        out = self.pssr_conv2(out4)
        return out
