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


''' Blocks; either simple, time conditioned, or time and a_t's conditioned'''

# can add batch norm and dropout
class DiffusionBlock(nn.Module):
    def __init__(self, 
                 nunits, 
                 dropout_rate, 
                 skip_connection, 
                 group_norm,
                 activation = nn.SiLU):
        super(DiffusionBlock, self).__init__()
        
        self.skip_connection = skip_connection # boolean
        self.act = activation(inplace=False)
        
        # for the moment, implementing as batch norm
        self.group_norm1 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.group_norm2 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.mlp_1 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm1)
        self.mlp_2 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm2)
        
    def forward(self, x: torch.Tensor):
        if self.skip_connection:
            x_skip = x
        x = self.act(self.mlp_1(x))
        x = self.mlp_2(x)
        if self.skip_connection:
            x= x + x_skip
        return self.act(x)
    

# can add batch norm and dropout
class DiffusionBlockTime(nn.Module):
    def __init__(self, 
                 nunits, 
                 time_embedding_size, 
                 dropout_rate, 
                 skip_connection, 
                 group_norm,
                 activation = nn.SiLU):
        super(DiffusionBlockTime, self).__init__()
        
        self.skip_connection = skip_connection # boolean
        self.act = activation(inplace=False)
        
        # for the moment, implementing as batch norm
        self.group_norm1 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.group_norm2 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.mlp_1 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm1)
        # remove dropout from embedding?
        self.t_proj = nn.Sequential(self.dropout, 
                                    nn.Linear(time_embedding_size, nunits), 
                                    self.act)
        self.mlp_2 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm2)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        if self.skip_connection:
            x_skip = x
        x = self.act(self.mlp_1(x))
        x += self.t_proj(t_emb)
        x = self.mlp_2(x)
        if self.skip_connection:
            x = x + x_skip
        return self.act(x)

# can add batch norm and dropout
class DiffusionBlockConditioned(nn.Module):
    def __init__(self, 
                 nunits, 
                 dropout_rate, 
                 skip_connection, 
                 group_norm,
                 time_emb_size = False, 
                 activation = nn.SiLU):
        super(DiffusionBlockConditioned, self).__init__()
        
        self.skip_connection = skip_connection # boolean
        self.act = activation(inplace=False)
        # self.act = torch.nn.functional.relu # nn.ReLU() # (inplace=False)
        # use ReLU activation, as a module
        # self.act = nn.ReLU(inplace=True)
        
        
        self.time = time_emb_size != False
        
        # for the moment, implementing as batch norm
        self.group_norm1 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.group_norm2 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.mlp_1 = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits), 
                                   self.group_norm1)
        # remove dropout from embedding?
        if self.time:
            self.t_proj = nn.Sequential(self.dropout, 
                                        nn.Linear(time_emb_size, nunits), 
                                        self.act)

        self.mlp_2 = nn.Sequential(self.dropout, 
                                   (nn.Linear(nunits, nunits)), 
                                   self.group_norm2)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        if self.skip_connection:
            x_skip = x
        x = self.act(self.mlp_1(x))
        if self.time:
            x += self.t_proj(t_emb) 
        x = self.mlp_2(x)
        if self.skip_connection:
            x = x + x_skip
        return self.act(x)
    
    
    
class DiffusionBlockConditionedMultChannels(nn.Module):
    def __init__(self, 
                 nunits, 
                 dropout_rate, 
                 skip_connection, 
                 group_norm,
                 nchannels,
                 time_emb_size = False, 
                 activation = nn.SiLU,
                 ):
        super(DiffusionBlockConditionedMultChannels, self).__init__()
        
        # create as much DiffusionBlockConditioned as channels
        self.diffusion_blocks = nn.ModuleList([
            DiffusionBlockConditioned(nunits, 
                                    dropout_rate, 
                                    skip_connection, 
                                    group_norm, 
                                    time_emb_size, 
                                    activation) 
            for _ in range(nchannels)]
        )
        
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # create a new channel per DiffusionBlockConditioned
        x = torch.stack([db(x, t_emb) for i, db in enumerate(self.diffusion_blocks)], dim=1)
        return x
    