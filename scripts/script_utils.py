from torch.utils.data import DataLoader
import os

from manage.files import FileHandler
from manage.data import get_dataset, CurrentDatasetInfo, Modality, StateSpace
from manage.logger import Logger
from manage.generation import GenerationManager
from manage.training import TrainingManager
from evaluate.EvaluationManager import EvaluationManager
from datasets import get_dataset
from manage.checkpoints import load_experiment, save_experiment
from manage.setup import _get_device, _optimize_gpu, _set_seed

from ddpm_init import init_method_ddpm, init_models_optmizers_ls, init_learning_schedule

from script_utils import *

import argparse



def initialize_experiment(p):
    
    # initialize gpu backend
    print('Initializing GPU configuration...')
    device = _get_device()
    p['device'] = device
    print('Device set to', device)

    _optimize_gpu(device=device)
    if p['seed'] is not None:
        _set_seed(p['seed'])

    print('Done')
    
    # initialize logger
    print('Initializing logger...', end='')
    logger = None # implement your own logger
    print('Done')

    print('Loading data...')
    dataset_files, test_dataset_files, modality, state_space, has_labels = get_dataset(p)
    # implement DDP later on
    data = DataLoader(dataset_files, 
                    batch_size=p['training']['batch_size'], 
                    shuffle=True, 
                    num_workers=p['training']['num_workers'])
    test_data = DataLoader(test_dataset_files,
                            batch_size=p['training']['batch_size'],
                            shuffle=True,
                            num_workers=p['training']['num_workers'])
    # set the current dataset info
    CurrentDatasetInfo.set_dataset_info(
                modality = modality, 
                state_space = state_space,
                has_labels=has_labels
    )
    print('Data modality:{}, state_space:{}, labels={}'.format(modality, state_space, has_labels))
    print('Done')


    is_image_dataset = (CurrentDatasetInfo.modality == Modality.IMAGE)
    print('Preparing evaluation directories...', end='')        
    # prepare the evaluation directories
    file_handler = FileHandler(
        exp_name=None,
        eval_name=None,
    )
    # for custom checkpoint name:
    # file_handler.exp_hash = lambda p : ...
    # file_handler.eval_name = lambda p : ...

    gen_data_path, real_data_path = file_handler.prepare_data_directories(
        p,
        dataset_files, 
        is_image_dataset= is_image_dataset
    )
    print('Done')

    # initialize models, optimizers and learning schedules. They are stored in dictionnaries, in case we need to manage multiple models.
    print('Initializing models, optimizers and learning schedules...')
    models, optimizers, learning_schedules = init_models_optmizers_ls(p)
    print('Done')


    # initialize geenrative method
    print('Initializing generative method...', end='')
    method = init_method_ddpm(p)
    print('Done')

    # intialize generation manager
    print('Initialzing generation manager...', end='')
    # these kwargs will be passed to the 'sample' function of the generative method
    gen_kwargs = p['eval'][p['method']]
    gen_manager = GenerationManager(method, 
                                dataloader=data, 
                                is_image = is_image_dataset,
                                **gen_kwargs)
    print('Done')

    # run evaluation on train or test data
    print('Initializing evaluation manager...', end='')
    eval = EvaluationManager(
            method=method,
            gen_manager=gen_manager,
            dataloader=data, # or test_data
            verbose=True, 
            logger = logger,
            data_to_generate = p['eval']['data_to_generate'],
            batch_size = p['eval']['batch_size'],
            modality = modality,
            state_space = state_space,
            gen_data_path=gen_data_path,
            real_data_path=real_data_path
    )
    print('Done')

    '''
    In dictionnary p['training'][p['method']]:
    - ema_rates, grad_clip will be passed to the training loop function 
    - other parameters will be passed to the 'training_losses' function of the generative method
    '''
    # kwargs goes to manager (ema_rates), train_loop (grad_clip), and eventually to training_losses (monte_carlo...)
    print('Initializing training manager...', end='')
    train_kwargs = p['training'][p['method']]
    trainer = TrainingManager(models,
                data,
                method,
                optimizers,
                learning_schedules,
                eval,
                logger=logger,
                p=p,
                dataset_with_labels=has_labels,
                eval_freq = p['run']['eval_freq'],
                checkpoint_freq = p['run']['checkpoint_freq'],
                **train_kwargs
                )
    print('Done')
    
    return trainer, logger, file_handler, models, optimizers, learning_schedules, method, eval, gen_manager



def print_dict(d, indent = 0):
    for k, v in d.items():
        if isinstance(v, dict):
            print('\t'*indent, k, '\t:')
            print_dict(v, indent + 1)
        else:
            print('\t'*indent, k, ':', v)
            
            
# These parameters should be changed for this specific run, before objects are loaded
def update_parameters_before_loading(p, args):
    
    if args.method is not None:
        p['method'] = args.method
    
    method = p['method']
    
    if args.epochs is not None:
        p['run']['epochs'] = args.epochs
    
    if args.eval is not None:
        p['run']['eval_freq'] = args.eval

    if args.check is not None:
        p['run']['checkpoint_freq'] = args.check
    
    if args.train_reverse_steps is not None:
        p[method]['reverse_steps'] = args.train_reverse_steps
    
    if args.set_seed is not None:
        p['seed'] = args.set_seed

    if args.random_seed is not None:
        p['seed'] = None

    if args.reverse_steps is not None:
        p['eval'][method]['reverse_steps'] = args.reverse_steps
        
    if args.inner_loop is not None:
        p['eval'][method]['inner_loop'] = args.inner_loop
   
    if args.deterministic:
        p['eval'][method]['deterministic'] = True

    if args.generate is not None:
        #assert False, 'NYI. eval_files are stored in some folder, and the prdc and fid functions consider all the files in a folder. So if a previous run had generated more data, there is a contamination. To be fixed'
        p['eval']['data_to_generate'] = args.generate
        assert args.generate <= p['eval']['real_data'], 'Must have more real data stored that number of data points to generate'
    
    # will do the neceassary changes after loading
    if args.lr is not None:
        p['optim']['lr'] = args.lr
    
    if args.lr_steps is not None:
        p['optim']['lr_steps'] = args.lr_steps
    
    if args.lr_schedule is not None:
        if args.lr_schedule == 'None':
            p['optim']['schedule'] = None
        else:
            p['optim']['schedule'] = args.lr_schedule

    # Data
    if args.dataset is not None:
        p['data']['dataset'] = args.dataset
    
    if args.nsamples is not None:
        p['data']['nsamples'] = args.nsamples
        
    if args.dimension is not None:
        p['data']['d'] = args.dimension
    
    # model architecture
    if args.arch is not None:
        p['model']['architecture'] = args.arch
    
    arch = p['model']['architecture']
    # MLP
    if arch == 'mlp':
        if args.blocks is not None:
            p['model']['mlp']['nblocks'] = args.blocks

        if args.units is not None:
            p['model']['mlp']['nunits'] = args.units

        if args.t_embedding_type is not None:
            p['model']['mlp']['time_emb_type'] = args.t_embedding_type

        if args.t_embedding_size is not None:
            p['model']['mlp']['time_emb_size'] = args.t_embedding_size

    # UNet
    if arch == 'unet':
        if args.model_type is not None:
            p['model']['unet']['model_type'] = args.model_type

        if args.attn_resolutions is not None:
            p['model']['unet']['attn_resolutions'] = args.attn_resolutions

        if args.channel_mult is not None:
            p['model']['unet']['channel_mult'] = args.channel_mult

        if args.dropout is not None:
            p['model']['unet']['dropout'] = args.dropout

        if args.model_channels is not None:
            p['model']['unet']['model_channels'] = args.model_channels

        if args.num_heads is not None:
            p['model']['unet']['num_heads'] = args.num_heads

        if args.num_res_blocks is not None:
            p['model']['unet']['num_res_blocks'] = args.num_res_blocks

        if args.learn_variance is not None:
            p['model']['unet']['learn_variance'] = args.learn_variance

    # Transformer
    if arch == 'transformer':
        if args.time_hidden_dim is not None:
            p['model']['transformer']['time_hidden_dim'] = args.time_hidden_dim

        if args.d_model is not None:
            p['model']['transformer']['d_model'] = args.d_model

        if args.nhead is not None:
            p['model']['transformer']['nhead'] = args.nhead

        if args.num_layers is not None:
            p['model']['transformer']['num_layers'] = args.num_layers

        if args.dim_feedforward is not None:
            p['model']['transformer']['dim_feedforward'] = args.dim_feedforward

        if args.transformer_dropout is not None:
            p['model']['transformer']['dropout'] = args.transformer_dropout
    
    if arch == 'vae':
        if args.vae_nunits is not None:
            p['model']['vae']['nunits'] = args.vae_nunits

        if args.vae_nblocks is not None:
            p['model']['vae']['nblocks'] = args.vae_nblocks

        if args.vae_latent_dim is not None:
            p['model']['vae']['latent_dim'] = args.vae_latent_dim

    # DMPM
    if args.Tf is not None:
        p['dmpm']['Tf'] = args.Tf

    if args.lambda_ is not None:  # Use `lambda_` because `lambda` is a reserved keyword in Python
        p['dmpm']['lambda_'] = args.lambda_
        
    if args.sampling is not None:
        p['eval']['dmpm']['sampling_algorithm'] = args.sampling

    if args.schedule is not None:
        p['eval']['dmpm']['schedule'] = args.schedule
        
    if args.inner_loop_schedule is not None:
        p['eval']['dmpm']['inner_loop_schedule'] = args.inner_loop_schedule

    # Training, DMPM
    if args.gamma is not None:
        p['training']['dmpm']['divide_by_gamma'] = args.gamma

    if args.mu is not None:
        p['training']['dmpm']['mu'] = args.mu

    if args.zeta is not None:
        p['training']['dmpm']['zeta'] = args.zeta

    if args.eta is not None:
        p['training']['dmpm']['eta'] = args.eta
        
    
    # dfm
    if args.dfm_corrector is not None:
        p['eval']['dfm']['corrector_sampler'] = True
        p['eval']['dfm']['adaptative'] = True

    return p


# change some parameters for the run.
# These parameters should act on the objects already loaded from the previous runs
def update_experiment_after_loading(
    p, 
    optimizers,
    learning_schedules,
    init_learning_schedule,
    args):
    # scheduler
    schedule_reset = False 
    if args.lr is not None:
        schedule_reset = True
        for optim in optimizers.values():
            for param_group in optim.param_groups:
                param_group['lr'] = args.lr
            p['optim']['lr'] = args.lr
    if args.lr_steps is not None:
        schedule_reset = True
        p['optim']['lr_steps'] = args.lr_steps
    if schedule_reset:
        for k, ls in learning_schedules.items():
            ls = init_learning_schedule(p, optimizers[k])
            learning_schedules[k] = ls # redundant?



# some additional logging 
def additional_logging(
    p,
    logger,
    trainer,
    fh,
    args):
    # logging job id
    if (logger is not None) and (args.job_id is not None):
        logger.log('job_id', args.job_id)
    
    # logging hash parameter
    if (logger is not None):
        logger.log('hash_parameter', fh.exp_name(p))
    
    # logging hash eval
    if (logger is not None):
        logger.log('hash_eval', fh.eval_name(p))
    
    # starting epoch and batch
    if (logger is not None):
        logger.log('starting_epoch', trainer.epochs)
        logger.log('starting_batch', trainer.total_steps)


# define and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # processes to choose from. Either diffusion, pdmp, or 'nf' to use a normal normalizing flow.
    parser.add_argument("--method", help='generative method to use', default=None, type=str)

    # EXPERIMENT parameters, specific to TRAINING
    parser.add_argument("--config", help='config file to use', type=str, required=True)
    parser.add_argument("--name", help='name of the experiment. Defines save location: ./models/name/', type=str, required=True)
    parser.add_argument('--epochs', help='epochs', default=None, type = int)
    parser.add_argument('-r', "--resume", help="resume existing experiment", action='store_true', default=False)
    parser.add_argument('--resume_epoch', help='epoch from which to resume', default = None, type=int)
    parser.add_argument('--eval', help='evaluation frequency', default=None, type = int)
    parser.add_argument('--check', help='checkpoint frequency', default=None, type = int)
    parser.add_argument('--n_max_batch', help='max batch per epoch (to speed up testing)', default=None, type = int)
    parser.add_argument('--train_reverse_steps', help='number of diffusion steps used for training', default=None, type = int)

    parser.add_argument('--set_seed', help='set random seed', default = None, type=int)
    parser.add_argument('--random_seed', help='set random seed to a random number', action = 'store_true', default=None)

    parser.add_argument('--log', help='activate logging to neptune', action='store_true', default=False)
    parser.add_argument('--job_id', help='slurm job id', default=None, type = str)

    # EXPERIMENT parameters, specific to EVALUATION
    parser.add_argument('--ema_eval', help='evaluate all ema models', action='store_true', default = False)
    parser.add_argument('--no_ema_eval', help='dont evaluate ema models', action='store_true', default = False)
    parser.add_argument('--generate', help='how many images/datapoints to generate', default = None, type = int)
    parser.add_argument('--reset_eval', help='reset evaluation metrics', action='store_true', default = False)
    
    parser.add_argument('--reverse_steps', help='choose number of reverse_steps', default = None, type = int)
    parser.add_argument('--inner_loop', help='Number M of inner loops', default = None, type=int)
    
    parser.add_argument('--deterministic', help='use deterministic sampling', default = False, action='store_true')
    parser.add_argument('--clip', help='use clip denoised (diffusion)', default = False, action='store_true')
    

    # DATA
    parser.add_argument('--dataset', help='choose specific dataset', default = None, type = str)
    parser.add_argument('--nsamples', help='choose the size of the dataset', default = None, type = str)
    parser.add_argument('--dimension', help='Choose data dimension', default = None, type = int)
    

    # OPTIMIZER
    parser.add_argument('--lr', help='reinitialize learning rate', type=float, default = None)
    parser.add_argument('--lr_steps', help='reinitialize learning rate steps', type=int, default = None)
    parser.add_argument('--lr_schedule', help='set learning rate schedule', type=str, default = None)

    # MODEL
    parser.add_argument('--arch', help='choose model architecture', default = None, type = str)
    
    # MLP
    parser.add_argument('--blocks', help='choose number of blocks in mlp', default = None, type = int)
    parser.add_argument('--units', help='choose number of units in mlp', default = None, type = int)
    parser.add_argument('--t_embedding_type', help='choose time embedding type', default = None, type = str)
    parser.add_argument('--t_embedding_size', help='choose time embedding size', default = None, type = int)

    # UNet
    parser.add_argument('--model_type', help='Specify UNet model type', default=None, type=str)
    parser.add_argument('--attn_resolutions', help='Set attention resolutions for UNet', default=None, nargs='+', type=int)
    parser.add_argument('--channel_mult', help='Set channel multipliers for UNet', default=None, nargs='+', type=int)
    parser.add_argument('--dropout', help='Set dropout rate for UNet', default=None, type=float)
    parser.add_argument('--model_channels', help='Specify number of model channels in UNet', default=None, type=int)
    parser.add_argument('--num_heads', help='Set number of attention heads in UNet', default=None, type=int)
    parser.add_argument('--num_res_blocks', help='Set number of residual blocks in UNet', default=None, type=int)
    parser.add_argument('--learn_variance', help='Specify if variance is learnable', default=None, type=bool)

    # Transformer
    parser.add_argument('--time_hidden_dim', help='Set hidden dimension for time embedding in transformer', default=None, type=int)
    parser.add_argument('--d_model', help='Set model dimension for transformer', default=None, type=int)
    parser.add_argument('--nhead', help='Set number of attention heads in transformer', default=None, type=int)
    parser.add_argument('--num_layers', help='Set number of transformer layers', default=None, type=int)
    parser.add_argument('--dim_feedforward', help='Set feedforward dimension in transformer', default=None, type=int)
    parser.add_argument('--transformer_dropout', help='Set dropout rate for transformer', default=None, type=float)

    # vae
    parser.add_argument('--vae_nunits', help='', default=None, type=int)
    parser.add_argument('--vae_nblocks', help='', default=None, type=int)
    parser.add_argument('--vae_latent_dim', help='', default=None, type=int)
    
    # DMPM
    parser.add_argument('--Tf', help='Set time horizon', default=None, type=float)
    parser.add_argument('--lambda_', help='Set intensity factor lambda', default=None, type=float)
    
    parser.add_argument('--sampling', help='choose sampling algorithm (default, denoise_renoise)', default=None, type=str)
    parser.add_argument('--schedule', help='set schedule for sampling at evaluation', default=None, type=str)
    parser.add_argument('--inner_loop_schedule', help='set inner loop schedule for sampling at evaluation', default=None, type=str)

    # DMPM, LOSS
    parser.add_argument('--gamma', help='Divide loss by gamma_t', default=None, action='store_true')
    parser.add_argument('--mu', help='Set mu in loss', default=None, type=float)
    parser.add_argument('--zeta', help='Set zeta in loss', default=None, type=float)
    parser.add_argument('--eta', help='Set eta in loss', default=None, type=float)
    
    
    # dfm
    parser.add_argument('--dfm_corrector', help='activate adaptative corrector sampler', default=None, action='store_true')
    
    
    # PARSE AND RETURN
    args = parser.parse_args()
    assert (args.no_ema_eval and args.ema_eval) == False, 'No possible evaluation to make'
    return args