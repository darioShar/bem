

import torch
import torch.nn as nn
from .unet_discrete import UNetModelDiscrete
from .unet import UNetModel
from .VAE import VAEGaussianBernoulli, VAESimple
from .MiniTransformer import TransformerForBits
from .MLP import MLPModel 
import torch.optim as optim
from transformers import get_scheduler


# for the moment, only unconditional models
def discrete_unet_model(p):
    channels = p['data']['channels']
    out_channels = channels
    p_model_unet = p['model']['unet']
    
    if (p['method'] in ['pdmp']):
        first_layer_embedding = False
    else:
        first_layer_embedding = True
    
    if (p['method'] in ['dmpm']):
        embedding_dim = 2
    else:
        embedding_dim = 3 # MD4 and DFM need a mask
        
    if (p['method'] == 'dfm'):
        output_dim = 3 # DFMs needs mask probability 
    elif (p['method'] == 'pdmp'):
        output_dim = 2
    else:
        output_dim = 1 # else, we only need a single probability
    
    model = UNetModelDiscrete(
            in_channels=channels,
            model_channels=p_model_unet['model_channels'],
            out_channels= out_channels,
            num_res_blocks=p_model_unet['num_res_blocks'],
            attention_resolutions=p_model_unet['attn_resolutions'],# tuple([2, 4]), # adds attention at image_size / 2 and /4
            dropout= p_model_unet['dropout'],
            channel_mult= p_model_unet['channel_mult'], # divides image_size by two at each new item, except first one. [i] * model_channels
            dims = 2, # for images
            num_classes= None,#(NUM_CLASSES if class_cond else None),
            use_checkpoint=False,
            num_heads=p_model_unet['num_heads'],
            num_heads_upsample=-1, # same as num_heads
            use_scale_shift_norm=True,
            use_sigmoid=True if output_dim==1 else False,
            first_layer_embedding=first_layer_embedding,
            embedding_dim= embedding_dim,
            output_dim = output_dim,
        )
    return model

# for the moment, only unconditional models
def unet_model(p):
    channels = p['data']['channels']
    out_channels = channels
    p_model_unet = p['model']
    
    model = unet.UNetModel(
            in_channels=channels,
            model_channels=p_model_unet['model_channels'],
            out_channels= out_channels,
            num_res_blocks=p_model_unet['num_res_blocks'],
            attention_resolutions=p_model_unet['attn_resolutions'],# tuple([2, 4]), # adds attention at image_size / 2 and /4
            dropout= p_model_unet['dropout'],
            channel_mult= p_model_unet['channel_mult'], # divides image_size by two at each new item, except first one. [i] * model_channels
            dims = 2, # for images
            num_classes= None,#(NUM_CLASSES if class_cond else None),
            use_checkpoint=False,
            num_heads=p_model_unet['num_heads'],
            num_heads_upsample=-1, # same as num_heads
            use_scale_shift_norm=True,
        )
    return model



def default_init_optimizer(p, model):
    # training manager
    optimizer = optim.AdamW(model.parameters(), 
                            lr=p['optim']['lr'], 
                            betas=(0.9, 0.999)) # beta_2 0.95 instead of 0.999
    return optimizer


def default_init_ls(p, optim):
    if p['optim']['schedule'] == None:
        return None
    
    if p['optim']['schedule'] == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size= p['optim']['lr_step_size'], 
                                                    gamma= p['optim']['lr_gamma'], 
                                                    last_epoch=-1)
    else: 
        lr_scheduler = get_scheduler(
            p['optim']['schedule'],
            # "cosine",
            # "cosine_with_restarts",
            optimizer=optim,
            num_warmup_steps=p['optim']['warmup'],
            num_training_steps=p['optim']['lr_steps'],
        )
    return lr_scheduler

