import os
from manage.files import FileHandler
from generative_methods.GenerativeLevyProcess import GenerativeLevyProcess


import torch
import torch.nn as nn
from models.unet_discrete import UNetModelDiscrete
from models.unet import UNetModel
from models.VAE import VAEGaussianBernoulli, VAESimple
from models.MiniTransformer import TransformerForBits
from models.MLP import MLPModel
from models import unet_model, discrete_unet_model
import torch.optim as optim
from transformers import get_scheduler

''' at any point during the program execution, will give information about the current dataset being used '''
from manage.data import CurrentDatasetInfo, Modality, StateSpace


''' This function should return an object which implements the following functions:

training_losses - returns the training losses of the model. Its arguments are:
    models - a list of trained models
    x_start - the starting point of the trajectory
    model_kwargs - a dictionary of keyword arguments to pass to the model
    **kwargs - additional keyword arguments

sample - returns a sample trajectory from the model. Its arguments are:
    models - a list of trained models
    shape - the shape of the data to generate
    **kwargs - additional keyword arguments
    
'''
def init_method_ddpm(p):
    chosen_gen_model = p['method']
    assert chosen_gen_model in ['dlpm'], f"In this implementation, chosen method should be in 'dlpm', got {chosen_gen_model}"
    method = GenerativeLevyProcess(alpha = p[chosen_gen_model]['alpha'],
                                device = p['device'],
                                reverse_steps = p[chosen_gen_model]['reverse_steps'],
                                rescale_timesteps = p[chosen_gen_model]['rescale_timesteps'],
                                isotropic = p[chosen_gen_model]['isotropic'],
                                model_mean_type = p[chosen_gen_model]['mean_predict'],
                                model_var_type = p[chosen_gen_model]['var_predict'],
                                scale = p[chosen_gen_model]['scale'],
                                input_scaling = p[chosen_gen_model]['input_scaling'],
    )
    return method



''' 
This function should return a neural network model object.
The model object can be initialized with the parameters in the dictionary p.
The model object should be moved to the device specified in p['device']

You can retrieve the modality of the data and the state space of the data using the CurrentDatasetInfo class.
The modality of the data can be accessed using CurrentDatasetInfo.modality
The state space of the data can be accessed using CurrentDatasetInfo.state_space
'''
def init_model(p):
    modality = CurrentDatasetInfo.modality
    state_space = CurrentDatasetInfo.state_space
    
    assert state_space == StateSpace.CONTINUOUS, 'only continuous state space supported in our example file'
    model = None
    arch = p['model']['architecture']
    print('Initializing model with architecture {}'.format(arch))
    if arch == 'mlp':
        model = MLPModel(p)
    elif arch == 'transformer':
        assert False, 'transformer model only implemented for discrete data'
        
    elif p['model']['architecture'] == 'vae':
        image_size = p['data']['image_size']
        model = VAESimple(
            shape = (1, image_size, image_size),
            device=p['device'],
            **p['model']['vae'])
        
    elif p['model']['architecture'] == 'unet':
        print('Using {} implementation'.format(p['model']['unet']['model_type']))
        if p['model']['unet']['model_type'] == 'ddpm':
            model = unet_model(p)
        else:
            raise ValueError('model type {} not recognized'.format(p['model']['unet']['model_type']))
    else:
        raise ValueError('model architecture {} not recognized'.format(p['model']['architecture']))
    
    assert model is not None, 'model could not be initialized with architecture = {}'.format(p['model']['architecture'])
    
    return model.to(p['device'])


''' 
This function should return an optimizer object.
The optimizer object can be initialized with the parameters in the dictionary p.
'''
def init_optimizer(p, model):
    optimizer = None
    if p['optim']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=p['optim']['lr'], betas=(0.9, 0.999)) # beta_2 0.95 instead of 0.999)
    elif p['optim']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=p['optim']['lr'])
    elif p['optim']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=p['optim']['lr'])
    else:
        raise ValueError('optimizer {} not recognized'.format(p['optim']['optimizer']))
    return optimizer


''' 
This function should return a learning schedule object.
The learning schedule object can be initialized with the parameters in the dictionary p.
'''
def init_learning_schedule(p, optim):
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

''' 
This function should return a dictionary of models, optimizers and learning schedules.
The keys of the dictionaries should be the names of the models, optimizers and learning schedules.
The values should be the corresponding objects.

We use this dictionnary structure because some generative methods require multiple models.
'''
def init_models_optmizers_ls(p):
    model = init_model(p)
    optimizer = init_optimizer(p, model)
    learning_schedule = init_learning_schedule(p, optimizer)
    
    models = {'default': model}
    optimizers = {'default': optimizer}
    learning_schedules = {'default': learning_schedule}
    return models, optimizers, learning_schedules


