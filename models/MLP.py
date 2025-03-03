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
    possible_time_embeddings = [
        'sinusoidal',
        'learnable',
        'one_dimensional_input'
    ]

    def __init__(self, p):
        super(MLPModel, self).__init__()

        # extract from param dict
        self.nfeatures =        p['data']['d']
        self.time_emb_type =    p['model']['mlp']['time_emb_type'] 
        self.time_emb_size =    p['model']['mlp']['time_emb_size']
        self.nblocks =          p['model']['mlp']['nblocks'] 
        self.nunits =           p['model']['mlp']['nunits']
        self.skip_connection =  p['model']['mlp']['skip_connection']
        self.group_norm =       p['model']['mlp']['group_norm']
        self.dropout_rate =     p['model']['mlp']['dropout_rate']
        self.softplus =         p['model']['mlp']['softplus']
        self.beta =             p['model']['mlp']['beta']
        self.threshold =        p['model']['mlp']['threshold']
        self.out_channel_mult = p['model']['mlp']['out_channel_mult']
        self.device =           p['device']
        
        # to be computed later depending on chosen architecture
        self.additional_dim =   0 

        assert self.time_emb_type in self.possible_time_embeddings
        
        # for dropout and group norm.
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.group_norm_in = nn.LayerNorm([self.nunits]) if self.group_norm else nn.Identity()
        # self.act = nn.SiLU(inplace=False)
        self.act = nn.SiLU(inplace=False)
        
        # manage time embedding type
        if self.time_emb_type == 'sinusoidal':
            self.time_emb = \
            SinusoidalPositionalEmbedding(self.diffusion_steps, 
                                              self.time_emb_size, self.device)
        elif self.time_emb_type == 'learnable':
            self.time_emb = nn.Linear(1, self.time_emb_size).to(self.device) #Embedding.LearnableEmbedding(1, self.time_emb_size, self.device)
        elif self.time_emb_type == 'one_dimensional_input':
            self.additional_dim += 1
        
        if self.time_emb_type != 'one_dimensional_input':
            # possibly, remove the mlp and just use the embedding
            self.time_mlp = nn.Sequential(self.time_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)
        
        self.linear_in =  nn.Linear(self.nfeatures + self.additional_dim, self.nunits)
        
        self.inblock = nn.Sequential(self.linear_in,
                                     self.group_norm_in, 
                                     self.act)
        
        self.midblocks = nn.ModuleList([DiffusionBlockConditioned(
                                            self.nunits, 
                                            self.dropout_rate, 
                                            self.skip_connection, 
                                            self.group_norm,
                                            time_emb_size = self.time_emb_size \
                                                if self.time_emb_type != 'one_dimensional_input'\
                                                else False,
                                            activation = nn.SiLU)
                                        for _ in range(self.nblocks)])
        
        if self.out_channel_mult > 1:
            self.outblock =DiffusionBlockConditionedMultChannels(
                        self.nunits, 
                        self.dropout_rate, 
                        self.skip_connection, 
                        self.group_norm,
                        self.out_channel_mult, # number of channels to create
                        time_emb_size = self.time_emb_size \
                            if self.time_emb_type != 'one_dimensional_input'\
                            else False,
                        activation = nn.SiLU
                        )
        else:
            self.outblock =DiffusionBlockConditioned(
                            self.nunits, 
                            self.dropout_rate, 
                            self.skip_connection, 
                            self.group_norm,
                            time_emb_size = self.time_emb_size \
                                if self.time_emb_type != 'one_dimensional_input'\
                                else False,
                            activation = nn.SiLU
                            )
        if self.out_channel_mult > 1:
            self.last_layer = LinearMultChannels(
                self.nunits, 
                self.nfeatures, 
                nchannels=3)
        else:
            self.last_layer = nn.Linear(self.nunits, self.nfeatures)
        
        
        
        # add a Softplus activation
        if self.softplus:
            self.act_softplus = nn.Softplus(beta=self.beta, threshold=self.threshold)
        
        

    def forward(self, x, timestep):
        
        x = x.float()

        timestep = timestep.unsqueeze(1) # create batch dimension
        
        inp = [x]
        # same but for time variable
        if self.time_emb_type == 'one_dimensional_input':
            inp += [timestep]
            # set to zero because we must feed a dummy to midblocks
            t = torch.zeros(size=timestep.size())
        else:
            t = self.time_mlp(timestep.to(torch.float32))
        
        # create channel dimension
        t = t.unsqueeze(1)
        
        # input
        val = torch.hstack(inp)
        # compute
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val, t)
        
        val_1 = self.outblock(val, t)
        val_1 = self.last_layer(val_1)        
        if self.softplus:
            val_1 = self.act_softplus(val_1)
        return val_1
            
        # duplicate last
        #val = val.expand(val.shape[0], 2*val.shape[1], *val.shape[2:])

    
    