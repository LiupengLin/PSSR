# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 15:22
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


from torch.utils.data import Dataset


class PSSRDataset(Dataset):
    def __init__(self, lr_train, hr_train):
        super(PSSRDataset, self).__init__()
        self.lr_train = lr_train
        self.hr_train = hr_train

    def __getitem__(self, index):
        batch_lr_train = self.lr_train[index]
        batch_hr_train = self.hr_train[index]
        return batch_lr_train.float(), batch_hr_train.float()

    def __len__(self):
        return self.lr_train.size(0)



