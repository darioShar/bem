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


CONFIG_PATH = './configs/'


    
def run_exp(config_path):
    args = parse_args()
    
    # Specify directory to save and load checkpoints
    checkpoint_dir = 'checkpoints'
    save_dir = os.path.join(checkpoint_dir, args.name)
    
    
    # open and get parameters from file
    p = FileHandler.get_param_from_config(config_path, args.config + '.yml')
    update_parameters_before_loading(p, args)

    trainer, logger, file_handler, models, optimizers, learning_schedules, method, eval, gen_manager = initialize_experiment(p)
    
    # load if necessary. Must be done here in case we have different hashes afterward
    if args.resume:
        load_experiment(
                p=p,
                trainer=trainer,
                file_handler = file_handler,
                save_dir=save_dir,
                checkpoint_epoch=args.resume_epoch if args.resume_epoch is not None else None,
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
        
    # run training
    def checkpoint_callback(checkpoint_epoch):
        print('saved files to', save_experiment(checkpoint_epoch=checkpoint_epoch))

    # run the training loop wuth parameters from the configuration file
    # specifying arguments here will overwrite the arguments obtained from the configuration file, for this training run
    trainer.train(
        total_epoch=p['run']['epochs'], 
        checkpoint_callback=checkpoint_callback,
        no_ema_eval=args.no_ema_eval, # if True, will not run evaluation with EMA models
        progress= p['run']['progress'], # if True, will print progress bar
        max_batch_per_epoch= args.n_max_batch, 
    )
    
    # in any case, save the final model
    save_experiment(p=p,
                    trainer = trainer,
                    fh = file_handler,
                    save_dir=save_dir,
                    checkpoint_epoch=p['run']['epochs'])
    
    # terminates logger 
    if logger is not None:
        logger.stop()



if __name__ == '__main__':
    run_exp(CONFIG_PATH)