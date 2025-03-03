from torch.utils.data import DataLoader
import os

from manage.files import FileHandler
from data.data import get_dataset, CurrentDatasetInfo, Modality, StateSpace
from manage.logger import Logger
from manage.generation import GenerationManager
from manage.training import TrainingManager
from evaluate.EvaluationManager import EvaluationManager
from manage.checkpoints import load_experiment, save_experiment
from manage.setup import _get_device, _optimize_gpu, _set_seed

from ddpm_init import init_method_ddpm, init_models_optmizers_ls, init_learning_schedule

from script_utils import *

import matplotlib.pyplot as plt
from manage.display import get_plot, get_animation


CONFIG_PATH = './configs/'
SAVE_ANIMATION_PATH = './animation'
SAVE_FIG_PATH = './figures'


def display_exp(config_path):
    args = parse_args()
    
    # Specify directory to save and load checkpoints
    checkpoint_dir = 'checkpoints'
    save_dir = os.path.join(checkpoint_dir, args.name)
    
    
    # open and get parameters from file
    p = FileHandler.get_param_from_config(config_path, args.config + '.yml')
    update_parameters_before_loading(p, args)

    trainer, logger, file_handler, models, optimizers, learning_schedules, method, eval, gen_manager = initialize_experiment(p)


    print('Loading latest model')
    load_experiment(
        p=p,
        trainer=trainer,
        file_handler = file_handler,
        save_dir=save_dir,
        checkpoint_epoch=None,
    )
    
    # update parameters after loading, like new optim learning rate...
    update_experiment_after_loading(p, 
        optimizers,
        learning_schedules,
        init_learning_schedule,
        args,
    )
    
    # log some additional information
    additional_logging(p,
        logger,
        trainer,
        file_handler,
        args
    ) 
    
    # print parameters to stdout
    print_dict(p)

    # some information
    run_info = [p['data']['dataset'], method.reverse_steps, trainer.total_steps]
    title = '{}, reverse_steps={}, training_steps={}'.format(*run_info[:3])
    
    select_ema_model = True 
    
    if select_ema_model:
        ema_model = None 
        mu = 0.99
        for ema_dict in trainer.ema_objects:
            if ema_dict['default'].mu == mu:
                ema_model = ema_dict['default']
                break
        if ema_model is None:
            raise ValueError('No EMA model with mu={} found'.format(mu))
        
        print('Using EMA model with mu={}'.format(mu))
        models = {'default': ema_model}
    
    # number of points to display
    nsamples = 20000 if args.generate is None else args.generate
    
    gen_manager.generate(models, 
                    nsamples=nsamples, 
                    reverse_steps=100, 
                    print_progression=True, 
                    get_sample_history=True)
    
    generated_samples = gen_manager.samples
    generated_samples_history = gen_manager.history
    
    data_samples = gen_manager.load_original_data(nsamples)
    
    # display plot and animation, for a specific model
    tmp_gen = generated_samples.clamp(-1.5, 1.5)
    tmp_data = data_samples.clamp(-1.5, 1.5)


    fig = get_plot(tmp_gen, tmp_data, figsize=(7, 7), s = 1)
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.title('Generated samples')
    # save plot
    path = os.path.join(SAVE_FIG_PATH, '_'.join([str(x) for x in run_info]))
    fig.savefig(path + '.png', bbox_inches='tight', dpi=300)
    print('Plot saved in {}'.format(path))
    plt.show() 
    
    
    # save animation
    anim = get_animation(method=method,
                     generated_data_history = generated_samples_history, 
                     original_data= data_samples, 
                     is_image = False, 
                     title='Generated samples',
                     figsize=(7, 7), 
                     s = 1)
    
    path = os.path.join(SAVE_ANIMATION_PATH, '_'.join([str(x) for x in run_info]))
    anim.save(path + '.mp4')
    print('Animation saved in {}'.format(path))
    # stops the thread from continuing
    plt.show()

    # close everything
    if logger is not None:
        logger.stop()


if __name__ == '__main__':
    display_exp(CONFIG_PATH)