# -*- coding: utf-8 -*- #
 
# ------------------------------------------------------------------
# File Name:        layers
# Author:           郑徐瀚宇
# Version:          ver0_1
# Created:          2023/11/21
# Description:      Architecture
# class List:       Reshape() -- Reshape tensor (to be used in nn.Sequential)
#                   SSL() -- VAE architecture with additional graph structure inference module
#                   TGCNConv() -- Multi-hop message passing along time 
#                   Predictor() -- Main architecture combining TGCN and plain weighted-GCN for down-stream tasks
# History:
#       <author>        <version>       <time>      <desc>
#       郑徐瀚宇         ver0_1          2023/11/21  None
#       郑徐瀚宇         ver0_2          2023/11/24  Parameterize Time-aggragating Matrix
# ------------------------------------------------------------------

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import utils as tg_utils
from model import utils
from torch import sparse
import model.layers as layers

class DSSLTS():
    """ Decoupled Semi-Supervised Time-Series Learning
    ---
    Methdos:
        __init__:
        ssl:
        predictor:
        mat_pow:
        cal_time_adj:
        to:
    ---
    Properties:
        sslParam:
        predParam:
    """
    def __init__(self, feature_dim, latent_dim, seq_length, if_img, input_dim, hidden_dim, class_dim, if_forec, beta, tau=3, lamb=0.2):
        """Initializing DSSTSL
        ---
        Parameters:
            feature_dim: Number of features of node
            latent_dim: Number of independent latent variables
            if_img(bool): If a node stands for a image
            input_dim: Input dimention of predictor
            output_dim: Output dimention of predictor
            class_dim: Number of classes
            tau: Message Passing Stride of Time
            lamb: Forgetting Coefficient
        """
        super(DSSLTS, self).__init__()
        # Learning Structure
        self.SSLnet = layers.SSL(feature_dim, latent_dim, if_img)

        # Predictor Modules
        self.predNet = layers.Predictor(input_dim, hidden_dim, class_dim, if_forec, if_img, beta, seq_length, tau=tau)
        self.time_adj = self.cal_time_adj(seq_length, tau, lamb)

    def ssl(self, TSdata):
        """Self-supervised structure learning (feedward)
        ---
        Parameters:
            TSdata: Time-series data
        """ 
        
        return self.SSLnet(TSdata) # loss of batch
        
    
    def predictor(self, TSdata):
        """Perdictor with hidden structure and time-structure
        ---
        Parameters:
            TSdata: Time-series data
        ---
        Return:
            output:
        """
        source = torch.tensor(range(TSdata.shape[1])).repeat(1, TSdata.shape[1])
        target = torch.tensor(range(TSdata.shape[1])).unsqueeze(1).repeat(1, TSdata.shape[1]).flatten().unsqueeze(0)
        edge = torch.cat((source, target), dim=0).to(TSdata.device) 
        edge_attr = None
        for i in range(TSdata.shape[0]):
            edge_attr = utils.ts_append(edge_attr, self.SSLnet.inference(TSdata[i], edge))
        # input = self.SSLnet.embedding(TSdata)
        
        output = self.predNet(TSdata, self.time_adj.to(TSdata.device), edge, edge_attr)

        return output
    
    def mat_pow(self, edge_index, eye_m, tau):
        """Matrix Power for tau times
        ---
        Parameters:
            edge_index:
            eye_m:
            tau:
        ---
        Return:
            result:
        """
        result = edge_index
        for i in range(tau - 1):
            result = result @ edge_index - result + eye_m
        return result

    def cal_time_adj(self, seq_length, tau, lamb):
        """Calculate time decay adjaciency
        ---
        Parameters:
            seq_length:
            tau:
            lamb:
        ---
        Return:
            time_adj
        """
        edge_t = torch.tensor([range(1, seq_length), range(0, seq_length-1)])
        # print(edge_t.shape)
        # Calculating Adjacency Matrix on Time
        edges = tg_utils.to_torch_coo_tensor(edge_t)
        edges = (1 - lamb) * edges
        eye_m = torch.tensor([(range(seq_length)), range(seq_length)])
        eye_m = tg_utils.to_torch_coo_tensor(eye_m)
        time_adj = eye_m + edges
        time_adj = self.mat_pow(time_adj, eye_m, tau)

        return time_adj

    def to(self, device):
        """Setting calculating device
        ---
        Parameters:
            device: calculating device
        """
        self.SSLnet.to(device)
        self.predNet.to(device)

    @property
    def sslParam(self):
        return self.SSLnet.parameters()
    
    @property
    def predParam(self):
        return self.predNet.parameters()


class TSLSTM(nn.Module):
    """LSTM for Time-series prediction
    ---
    Methods:
        __init__:
        forward:
    """
    def __init__(self, input_dim, hidden_dim,  class_dim, num_layers, if_bi):
        """Initializing TSLSTM
        ---
        Parameters:
            input_dim: Number of input data features
            hidden_dim: Number of hidden features
            class_dim:
            num_layers:
            if_bi: If using bidirectional LSTM
        """
        super(TSLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bool(if_bi)) # [batch_size, seq_length, input_dim] -> [batch_size, seq_length, hidden_dim]
        self.fc = nn.Linear(hidden_dim, class_dim)

    def forward(self, TSdata):
        """Feed forward
        ---
        Parameters:
            TSdata:
        """
        output, (_, _) = self.lstm(TSdata)
        output = self.fc(output[:, -1, :])
        return F.softmax(output, dim=1)


if __name__ == '__main__':
    print(torch.tensor(range(10)).repeat(1, 10).squeeze(0))
    print(torch.tensor(range(10)).unsqueeze(1).repeat(1, 10).flatten())
    print(torch.tensor([range(0, 9), range(1, 10)]).shape)
    
