# -*- coding: utf-8 -*- #
 
# ------------------------------------------------------------------
# File Name:        layers
# Author:           郑徐瀚宇
# Version:          ver0_1
# Created:          2023/11/21
# Description:      Layers of neural networks
# class List:       Reshape() -- Reshape tensor (to be used in nn.Sequential)
#                   SSL() -- VAE architecture with additional graph structure inference module
#                   TGCNConv() -- Multi-hop message passing along time 
#                   Predictor() -- Main architecture combining TGCN and plain weighted-GCN for down-stream tasks
# History:
#       <author>        <version>       <time>      <desc>
#       郑徐瀚宇         ver0_1          2023/11/21  None
# ------------------------------------------------------------------


import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import utils
from torch_geometric.nn import GCNConv
from torch_geometric.nn import pool
from torch.distributions import Normal, kl_divergence
import torchvision.models as models

class Reshape(nn.Module):
    """To Use Reshape in nn.Sequential
    ---
    """
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.contiguous().view((x.size(0),)+self.shape)


class SSL(nn.Module):
    """Modified AutoEncoder for Structure Learning
    ---
    Methods:
        __init__:
        sample: Reparameterization
        forward: Forward Calculation
        inference: Calculate 
    """
    def __init__(self, feature_dim, latent_dim, if_img=False):
        """Initializing
        ---
        Parameters:
            feature_dim: Number of features of node
            latent_dim: Number of independent latent variables
            if_img(bool): If a node stands for a image
        """
        super(SSL, self).__init__()

        self.if_img = if_img
        self.latent_dim = latent_dim
        # Encoder
        if if_img:
            # Convolution encoder for image
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                Reshape(16 * 16 * 32)
            ) # [128, 128, 3] -> [16, 16, 32]

            self.fc = nn.Linear(16 * 16 * 32, 2 * latent_dim)

            # Deconvolution decoder for image (Using TransConv) 
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16 * 16 * 32),
                nn.ReLU(),
                Reshape(32, 16, 16),
                nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
                nn.ReLU()
            )

        else:
            self.encoder = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            ) # [num_edges, feature_dim] -> [num_edges, 128]

            self.fc = nn.Linear(128, 2 * latent_dim)
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim)
            ) # [num_edges, latent_dim] -> [num_edges, feature_dim]

    def sample(self, mu, log_std):
        """Reparemeterizaion sampling
        ---
        Parameters:
            mu:
            log_std:
        ---
        Return:
        """
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        """Feed forward
        ---
        Parameters:
            x: Nodes [batch_size, num_nodes, feature_dim]
            neigh: Mean aggregated neighborhoods of center node 
        Return:
            loss: Reconstruction loss
        """
        x = x.reshape(-1, 3, 128, 128) 
        hidden = self.encoder(x) # [batch_size, num_nodes, 128]
        mu = self.fc(hidden)[:,:self.latent_dim] # [batch_size, num_nodes, latent_dim]
        log_std = self.fc(hidden)[:,self.latent_dim:] # [batch_size, num_nodes, latent_dim]
        z = self.sample(mu, log_std) # [batch_size, num_nodes, latent_dim]
        output = self.decoder(z) # [batch_size, num_nodes, feature_dim]
        return F.mse_loss(output, x) + 0.5 * torch.sum(mu.pow(2) + log_std.exp() - log_std - 1)
    
    def inference(self, x, edge_index):
        """Calculate Adjacent Weights
        ---
        Parameters:
            x: Nodes
        """
        x = x.squeeze(0)
        hidden = self.encoder(x) # [num_nodes, 128]
        mu = self.fc(hidden)[:,:self.latent_dim] # [num_nodes, latent_dim]
        log_std = self.fc(hidden)[:,self.latent_dim:] # [num_nodes, latent_dim]
        mu_source = mu[edge_index[0], :] # [num_edges, latent_dim]
        std_source = torch.exp(0.5 * log_std)[edge_index[0], :] # [num_edges, latent_dim]
        distri_source = Normal(mu_source, std_source) # [num_edges, latent_dim]
        mu_target = mu[edge_index[1], :] # [num_edges, latent_dim]
        std_target = torch.exp(0.5 * log_std)[edge_index[1], :] # [num_edges, latent_dim]
        distri_target = Normal(mu_target, std_target) # [num_edges, latent_dim] 
        adj = 1 / (kl_divergence(distri_source, distri_target).mean(dim=1, keepdim=False) + 1) # [num_edges] 

        return adj # [num_edges]

    def embedding(self, x):
        """Embed nodes to hidden space
        ---
        Parameters:
            x: Nodes
            edge_index:full connected edges
        """
        hidden = self.encoder(x) # [batch_size, num_nodes, 128]
        
        return hidden


class TGCNConv(nn.Module):
    """ Forward Memoriable Message Passing for Time Series Graph Data
    ---
    Methods:
        __init__:
        forward:
    """
    def __init__(self, input_dim, output_dim, tau):
        super(TGCNConv, self).__init__()
        self.tau = tau
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_adj):
        """ Feed Forward on Time
        ---
        Parameters:
            x: Representation of nodes
            time_adj: Sparse adjacent matrix of time relation
        """
        # Message Passing
        output = (time_adj @ self.fc(x)) / self.tau
        return output
    
    
class Predictor(nn.Module):
    """Predictor of DSSL
    ---
    Methods:
        __init__:
        forward:
    """
    def __init__(self, input_dim, hidden_dim, class_dim, if_forec, if_img, beta, seq_length, tau=3):
        """Initialing Predictor

        ---
        Parameters:
            input_dim: Input dimention of predictor
            output_dim: Output dimention of predictor
            class_dim: Number of classes
            if_forec: If forecasting task
            tau: Message Passing Stride of Time
            lamb: Forgetting Coefficient
        """
        super(Predictor, self).__init__()
        self.if_img = if_img
        if if_img:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                Reshape(16 * 16 * 32),
                nn.Linear(16 * 16 * 32, 128),
                nn.ReLU(),
            ) # [128, 128, 3] -> [16, 16, 32]
            self.TSconv1 = TGCNConv(128, hidden_dim, tau=tau)
            self.TSconv2 = TGCNConv(hidden_dim, class_dim, tau=tau)

            self.GCNconv1 = GCNConv(128, hidden_dim)
            self.GCNconv2 = GCNConv(hidden_dim, class_dim)
        else:
            self.TSconv1 = TGCNConv(input_dim, hidden_dim, tau=tau)
            self.TSconv2 = TGCNConv(hidden_dim, class_dim, tau=tau)

            self.GCNconv1 = GCNConv(input_dim, hidden_dim)
            self.GCNconv2 = GCNConv(hidden_dim, class_dim)

        # self.fc = nn.Linear(seq_length, 1)

        self.if_forec = if_forec
        self.class_dim = class_dim
        self.beta = beta

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, TSdata, time_adj, edge, edge_attr):
        """Feed forward:
        ---
        Parameters:
            TSdata:
            time_adj:
            edge:
            edge_attr:
        ---
        Return:
        """
        hidden = None
        for i in range(TSdata.shape[0]):
            if self.if_img:
                input = self.encoder(TSdata[i]).squeeze(0)
            else:
                input = TSdata[i]
            # print(TSdata[0].shape, edge[0].shape, edge_t[0].shape, edge_attr.shape)
            hidden_t = self.TSconv1(input, time_adj) # [batch_size, num_nodes, hidden_dim]
            hidden_g = self.GCNconv1(input, edge_index=edge, edge_weight=edge_attr[i]) # [batch_size, num_nodes, hidden_dim]

            # hidden_tmp =hidden_g
            # hidden_tmp = F.relu(hidden_t + self.beta * hidden_g) # [batch_size, num_nodes, hidden_dim]
            # hidden_tmp = F.relu(self.beta * hidden_g) # [batch_size, num_nodes, hidden_dim]

            hidden_t = self.dropout(self.TSconv2(hidden_t, time_adj)) # [batch_size, num_nodes, class_dim]
            hidden_g = self.dropout(self.GCNconv2(hidden_g, edge_index=edge, edge_weight=edge_attr[i])) # [batch_size, num_nodes, class_dim]
            
            # hidden = torch.cat((hidden, hidden_g.unsqueeze(0)), dim=0)
            hidden = utils.ts_append(hidden, (hidden_t +  self.beta * hidden_g))
            # hidden = torch.cat((hidden, (self.beta * hidden_g).unsqueeze(0)), dim=0)

        if self.if_forec:
            return hidden.squeeze(2) # [batch_size, num_nodes, class_dim]
        else:   
            return F.softmax(hidden.sum(dim=1, keepdim=False), dim=1) # [batch_size, class_dim]

