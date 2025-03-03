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


CONFIG_PATH = './configs/'

def eval_exp(config_path):
    args = parse_args()
    
    # Specify directory to save and load checkpoints
    checkpoint_dir = 'checkpoints'
    save_dir = os.path.join(checkpoint_dir, args.name)
    
    
    # open and get parameters from file
    p = FileHandler.get_param_from_config(config_path, args.config + '.yml')
    update_parameters_before_loading(p, args)
    

    trainer, logger, file_handler, models, optimizers, learning_schedules, method, eval, gen_manager = initialize_experiment(p)

    if args.reset_eval:
        print('Resetting eval dictionnary')
        load_experiment(
                p=p,
                trainer=trainer,
                file_handler = file_handler,
                save_dir=save_dir,
                checkpoint_epoch=args.resume_epoch if args.resume_epoch is not None else None,
                )
        trainer.eval.reset(keep_losses=True, keep_evals=False)
        # in any case, save the final model
        save_experiment(p=p,
                        trainer = trainer,
                        fh = file_handler,
                        save_dir=save_dir,
                        checkpoint_epoch=p['run']['epochs'],
                        files = 'eval')
        print('Eval dictionnary reset and saved.')
        
    # log some additional information
    additional_logging(p,
        logger,
        trainer,
        file_handler,
        args
    ) 
    
    # print parameters to stdout
    print_dict(p)
    
    for epoch in range(args.eval, args.epochs + 1, args.eval):
        print('Evaluating model at epoch {}'.format(epoch))
        load_experiment(
                p=p,
                trainer=trainer,
                fh = file_handler,
                save_dir=save_dir,
                checkpoint_epoch=epoch
                )
        # evalute ema models
        trainer.evaluate(evaluate_emas=False)
        trainer.evaluate(evaluate_emas=True)
        paths = save_experiment(
            p=p,
            trainer = trainer,
            fh = file_handler,
            save_dir = save_dir,
            files=['eval', 'param'], 
            new_eval_subdir=True, 
            checkpoint_epoch=epoch)
        print('Saved (model, eval, param) in ', paths)
    
    # terminates logger 
    if logger is not None:
        logger.stop()


if __name__ == '__main__':
    eval_exp(CONFIG_PATH)