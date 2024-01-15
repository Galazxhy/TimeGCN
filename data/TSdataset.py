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
#       郑徐瀚宇         ver1_0          2023/01/10  Floatation Dataset
# ------------------------------------------------------------------


import os
import torch
import pandas as pd
import numpy as np
import cv2 as cv
import random
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
        self.seq_length = seq_length

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
            self.imgs = None
            self.labels = None
            seq_idxs = os.listdir(data_root + '/imgs')
            for seq_idx in seq_idxs:
                img = cv.imread(self.data_root+'/imgs/'+seq_idx+'/30.jpg')
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, (256, 256))
                img = img / 255.0
                self.imgs = utils.np_append(self.imgs, img)
                label_df = pd.read_csv(data_root+'/labels/'+seq_idx+'.csv')
                self.labels = utils.np_append(self.labels, np.array(label_df['label'][60]))

            self.labels = utils.lb_normalize(self.labels)
            self.allTestData = np.arange(0, 600, 1)
            rand_idx = np.random.choice(600, 600, False)
            self.train_data = rand_idx[:500]
            self.test_data = rand_idx[500:]
            self.mask = (np.random.rand(self.imgs.shape[0])>0.4)

        elif data_root.split('/')[-1] == 'electricity':
            self.in_memory = True

            datas = pd.read_csv(data_root+'/electricity.csv').values()
            self.features = None
            self.labels = None   

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
                return self.train_data[index], -1, -1, self.train_label[index]
            elif self.mode == 'test':
                return self.test_data[index], -1, -1, self.test_label[index]
        else:
            if self.mode == 'train':
                seq_start = self.train_data[index]
                seq = self.imgs[seq_start:seq_start+self.seq_length]
                seq_label = self.labels[seq_start:seq_start+self.seq_length]
                seq_mask = self.mask[seq_start:seq_start+self.seq_length]
                return torch.tensor(seq, dtype=torch.float).permute(0, 3, 1, 2), torch.tensor(seq_label, dtype=torch.float), torch.tensor(seq_mask), -1

            elif self.mode == 'test':
                seq_start = self.test_data[index]
                seq = self.imgs[seq_start:seq_start+self.seq_length]
                seq_label = self.labels[seq_start:seq_start+self.seq_length]
                seq_mask = np.zeros(self.seq_length, dtype=bool)
                seq_mask[self.seq_length-1] = True
                return torch.tensor(seq, dtype=torch.float).permute(0, 3, 1, 2), torch.tensor(seq_label, dtype=torch.float), torch.tensor(seq_mask), -1
            
            elif self.mode == 'seq_test':
                seq_start = self.allTestData[index]
                seq = self.imgs[seq_start:seq_start+self.seq_length]
                seq_label = self.labels[seq_start:seq_start+self.seq_length]
                seq_mask = np.zeros(self.seq_length, dtype=bool)
                seq_mask[self.seq_length-1] = False
                return torch.tensor(seq, dtype=torch.float).permute(0, 3, 1, 2), torch.tensor(seq_label, dtype=torch.float), torch.tensor(seq_mask), -1



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
        self.seq_test_dataset = TSData(data_root, seq_length, 'seq_test')
        
        self.batch_size = batch_size
    
    @property
    def getTrainDataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, drop_last=True)
    
    @property
    def getTestDataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, drop_last=True)
    
    @property
    def getAllTestDataloader(self):
        return DataLoader(self.seq_test_dataset, self.batch_size, shuffle=False)

    @property
    def getInputShape(self):
        return self.train_dataset[0][0].shape

