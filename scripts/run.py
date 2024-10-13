import os
import Experiments as Exp
import Logger as Logger
from utils_exp import *
from script_utils import *

# import/define your own functions here for method and model initialization, logging, etc.

# path to the config files
CONFIG_PATH = 'configs/'

def run_exp(config_path):
    args = parse_args()
    # open and get parameters from file
    p = FileHandler.get_param_from_config(config_path, args.config + '.yml')

    update_parameters_before_loading(p, args)

    # create experiment object. Specify directory to save and load checkpoints, experiment parameters, and potential logger object
    checkpoint_dir = os.path.join('models', args.name)
    # the ExpUtils class specifies how to hash the parameter dict, and what and how to initiliaze methods and models
    exp = Exp.Experiment(checkpoint_dir=checkpoint_dir, 
                        p=p,
                        logger = ...,
                        exp_hash= None, # will use default function
                        eval_hash=None, # will use default function
                        init_method_by_parameter= ...,
                        init_models_by_parameter= ...,
                        reset_models= ...)


    # load if necessary. Must be done here in case we have different hashes afterward
    if args.resume:
        if args.resume_epoch is not None:
            exp.load(epoch=args.resume_epoch)
        else:
            exp.load()
    else:
        exp.prepare()
    
    update_experiment_after_loading(exp, args) # update parameters after loading, like new optim learning rate...
    additional_logging(exp, args) # log some additional information
    exp.print_parameters() # print parameters to stdout
    
    # run the experiment
    exp.run(progress= p['run']['progress'],
            max_batch_per_epoch= args.n_max_batch, # to speed up testing
            no_ema_eval=args.no_ema_eval, # to speed up testing
        )
    
    # in any case, save last models.
    print(exp.save(curr_epoch=p['run']['epochs']))
    
    # terminates everything (e.g., logger etc.)
    exp.terminate()


if __name__ == '__main__':
    run_exp(CONFIG_PATH)