# -*- coding: utf-8 -*- #
 
# ------------------------------------------------------------------
# File Name:        TSdataset
# Author:           郑徐瀚宇
# Version:          ver0_1
# Created:          2023/11/21
# Description:      Create Dataset & Dataloader
# Class List:       TSData() -- Initialize nn.Dataset type dataset
#                   GetData() -- Initialize nn.DataLoader type dataloader, offer iterators to get data
# History:
#       <author>        <version>       <time>      <desc>
#       郑徐瀚宇         ver0_1          2023/11/21  None
# ------------------------------------------------------------------


import os
import torch
import pandas as pd
import numpy as np
import cv2 as cv
from model import utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class TSData(Dataset):
    """Time series dataset
    ---
    Methods:
        __init__:
        __len__: Get length of dataset
        __getitem__: Get data by index 
    """
    def __init__(self, data_root, seq_length, mode):
        """
        Parameters:
            data_root: Root dir of data
            seq_length: Sample length of sequence
            mode: Train or test mode
        """
        super(TSData, self).__init__()
        self.mode = mode
        self.data_root = data_root

        if data_root.split('/')[-1] == 'UEA':
            self.in_memory = True
            # Preprocess UEA dataset
            files = os.listdir(data_root)
            train_csv = [file  for file in files if file.split('_')[1].split('.')[0] == 'TRAIN']
            test_csv = [file for file in files if file.split('_')[1].split('.')[0] == 'TEST']

            # Train data
            train_data = []
            for csv in train_csv:
                data = pd.read_csv(data_root+'/'+csv).to_numpy()
                train_data.append(data)
            self.train_data = torch.tensor(np.array(train_data).transpose(1, 2, 0), dtype=torch.float)
            self.train_data = (self.train_data - self.train_data.min(dim=0)[0]) / (self.train_data.max(dim=0)[0] - self.train_data.min(dim=0)[0])
            self.train_label = torch.tensor(pd.read_csv(data_root+'/'+'train_label.csv').to_numpy(), dtype=torch.float).squeeze(1) 

            # Test data
            test_data = []
            for csv in test_csv:
                data = pd.read_csv(data_root+'/'+csv).to_numpy()
                test_data.append(data)

            self.test_data = torch.tensor(np.array(test_data).transpose(1, 2, 0), dtype=torch.float)
            self.test_data = (self.test_data - self.test_data.min(dim=0)[0]) / (self.test_data.max(dim=0)[0] - self.test_data.min(dim=0)[0])
            self.test_label = torch.tensor(pd.read_csv(data_root+'/'+'test_label.csv').to_numpy(), dtype=torch.float).squeeze(1)

        elif data_root.split('/')[-1] == 'Floatation':
            # Preprocessing floatation dataset
            self.in_memory = False
            seq_idxs = os.listdir(data_root + '/imgs')
            labels = []
            for seq_idx in seq_idxs:
                labels.append(pd.read_csv(data_root + '/labels/' + seq_idx + '.csv'))
            train_num = int(len(seq_idxs) * 0.8)
            self.train_data = seq_idxs[:train_num]
            self.train_label = labels[:train_num]
            self.test_data = seq_idxs[train_num:]
            self.test_label = labels[train_num:]
    
    def __len__(self):
        """Get length
        """
        if self.in_memory:
            if self.mode == 'train':
                return self.train_data.shape[0]
            elif self.mode == 'test':
                return self.test_data.shape[0]
        else:
            if self.mode == 'train':
                return len(self.train_data)
            elif self.mode == 'test':
                return len(self.test_data)
    
    def __getitem__(self, index):
        """Get item
        """
        if self.in_memory:
            if self.mode == 'train':
                return self.train_data[index], -1, self.train_label[index]
            elif self.mode == 'test':
                return self.test_data[index], -1, self.test_label[index]
        else:
            if self.mode == 'train':
                seq = self.train_data[index]
                label_csv = self.train_label[index]
                imgs = None
                labels = None
                items = os.listdir(self.data_root + '/imgs/' + seq)
                items.sort(key = lambda x:int(x[:-4]))
                for item in items:
                    img = cv.imread(self.data_root+'/imgs/'+seq+'/'+item)
                    img = cv.resize(img, (128, 128))
                    img = img / 255.0
                    imgs = utils.np_append(imgs, img)
                    labels = utils.np_append(labels, np.array(label_csv['label'][int(item.split('.')[0])]))
                labels_mask = np.array(labels != -1)
                ySL =torch.stack((torch.tensor(labels, dtype=torch.float), torch.tensor(labels_mask, dtype=torch.bool)))
                return torch.tensor(imgs, dtype=torch.float).permute(0, 3, 1, 2), ySL, -1

            elif self.mode == 'test':
                seq = self.test_data[index]
                label_csv = self.test_label[index]
                imgs = None
                labels = None
                items = os.listdir(self.data_root + '/imgs/' + seq)
                items.sort(key = lambda x:int(x[:-4]))
                for item in items:
                    img = cv.imread(self.data_root+'/imgs/'+seq+'/'+item)
                    img = cv.resize(img, (128, 128))
                    img = img / 255.0
                    imgs = utils.np_append(imgs, img)
                    labels = utils.np_append(labels, np.array(label_csv['label'][int(item.split('.')[0])]))
                labels_mask = np.array(labels != -1)
                ySL = np.stack((labels, labels_mask))
                return torch.tensor(imgs, dtype=torch.float).permute(0, 3, 1, 2), torch.tensor(ySL, dtype=torch.float), -1


class GetData():
    """Preprocess data and construct iterator of dataset
    ---
    Methods:
        __init__:
    ---
    Functions:
        @property
        getTrainDataloader: Get train dataloader
        getValidDataloader: Get validation dataloader
        getTestDataloader: Get test dataloader
        getInputShape: Get x shape
    """
    def __init__(self, data_root, batch_size, seq_length, mode):
        """Initializing GetData
        ---
        Parameter:
            data_root: Root dir of data
            batch_size: Size of each batch
            seq_length: Sample length of sequence
            mode: Train, valid or test mode
        """
        self.train_dataset = TSData(data_root, seq_length, 'train')
        self.test_dataset = TSData(data_root, seq_length, 'test')
        
        self.batch_size = batch_size
    
    @property
    def getTrainDataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, drop_last=True)
    
    @property
    def getTestDataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, drop_last=True)

    @property
    def getInputShape(self):
        return self.train_dataset[0][0].shape

