import os
import numpy as np
import torch
import scipy

import torch.nn as nn
import torch.optim as optim
import shutil

from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


from .DiffusionBlocks import DiffusionBlockConditioned, DiffusionBlockConditionedMultChannels
from .Embeddings import SinusoidalPositionalEmbedding



class LinearMultChannels(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 nchannels,
                 ):
        super(LinearMultChannels, self).__init__()
        
        # create as much DiffusionBlockConditioned as channels
        self.linear_nn = nn.ModuleList([
            nn.Linear(in_dim, out_dim) 
            for _ in range(nchannels)]
        )
        
        
    def forward(self, x: torch.Tensor):
        # create a new channel per DiffusionBlockConditioned
        x = torch.stack([lin(x[:, i, ...]) for i, lin in enumerate(self.linear_nn)], dim=1)
        return x
    


#Can predict gaussian_noise, stable_noise, anterior_mean
class MLPModel(nn.Module):

    def __init__(self, p):
        super(MLPModel, self).__init__()

        # extract from param dict
        self.nfeatures =        p['data']['d']
        self.time_emb_size =    p['model']['mlp']['time_emb_size']
        self.nblocks =          p['model']['mlp']['nblocks'] 
        self.nunits =           p['model']['mlp']['nunits']
        self.skip_connection =  p['model']['mlp']['skip_connection']
        self.layer_norm =       p['model']['mlp']['layer_norm']
        self.dropout_rate =     p['model']['mlp']['dropout_rate']
        self.learn_variance =   p[ p['method'] ]['learn_variance']
        self.device =           p['device']
        self.num_classes =      p['data']['num_classes'] if p[p['method']]['conditional'] else None
        
        
        # for dropout and group norm.
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layer_norm_in = nn.LayerNorm([self.nunits]) if self.layer_norm else nn.Identity()
        self.act = nn.SiLU(inplace=False)
        
        
        
        self.time_emb = nn.Linear(1, self.time_emb_size) #Embedding.LearnableEmbedding(1, self.time_emb_size, self.device)
        self.time_mlp = nn.Sequential(self.time_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)
        
        if self.num_classes is not None:
            self.y_emb = nn.Embedding(self.num_classes + 1, self.time_emb_size)
            self.y_mlp = nn.Sequential(self.y_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size),
                                      self.act)
        
        self.linear_in =  nn.Linear(self.nfeatures, self.nunits)
        
        self.inblock = nn.Sequential(self.linear_in,
                                     self.layer_norm_in, 
                                     self.act)
        
        self.midblocks = nn.ModuleList([DiffusionBlockConditioned(
                                            self.nunits, 
                                            self.dropout_rate, 
                                            self.skip_connection, 
                                            self.layer_norm,
                                            time_emb_size = self.time_emb_size,
                                            activation = nn.SiLU)
                                        for _ in range(self.nblocks)])
        
        # add one conditioned block and one MLP for both mean and variance computation
        self.outblocks_mean_cond_mlp = DiffusionBlockConditioned(
                                                self.nunits, 
                                                self.dropout_rate, 
                                                self.skip_connection, 
                                                self.layer_norm,
                                                time_emb_size = self.time_emb_size,
                                                activation = nn.SiLU)
        self.outblocks_mean_ff = nn.Linear(self.nunits, self.nfeatures)
            
        if self.learn_variance:
            self.outblocks_var_mlp =  DiffusionBlockConditioned(
                                                self.nunits, 
                                                self.dropout_rate, 
                                                self.skip_connection, 
                                                self.layer_norm,
                                                time_emb_size = self.time_emb_size,
                                                activation = nn.SiLU)
            self.outblocks_mvar_ff = nn.Linear(self.nunits, self.nfeatures)
        
        
    def forward(self, x, timestep, y = None):
        
        t = timestep.unsqueeze(-1).unsqueeze(-1) # add batch dim and channel dim
        t = self.time_mlp(t.to(torch.float32))
        
        if (self.num_classes is not None) and (y is not None):
            y = self.y_mlp(y)
            y = y.squeeze(1) # embedding layers add an extra channel dimension
            t = t+ y
        
        # input
        val = x
        
        # input block
        val = self.inblock(val)
        
        # midblocks
        for midblock in self.midblocks:
            val = midblock(val, t)
        
        # output blocks
        val_mean = self.outblocks_mean_cond_mlp(val, t)
        val_mean = self.outblocks_mean_ff(val_mean)
        
        # if we learn variance
        if not self.learn_variance:
            return val_mean

        val_var = self.outblocks_var[0](val, t)
        val_var = self.outblocks_var[1](val_var)
        
        return torch.concat([val_mean, val_var], dim = 1) # concat on channels dim
        

    
    