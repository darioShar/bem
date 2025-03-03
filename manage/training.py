import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
from tqdm import tqdm
from .utils_ema import EMAHelper
#Wrapper around training and evaluation functions

def compute_layer_norms(model):
    layer_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad:  # Ensure the parameter is trainable
            norm = torch.norm(param)  # Frobenius norm by default
            # layer_norms.append((name, norm))
            layer_norms.append(norm)
    return torch.stack(layer_norms).detach()


class TrainingManager:
    
    def __init__(self, 
                 models, # dictionnary with model name as key. Must contain 'default' model
                 data,
                 method, 
                 optimizers,
                 learning_schedules,
                 eval,
                 logger = None,
                 ema_rates = None,
                 reset_models = None,
                 p = None,
                 **kwargs):
        self.epochs = 0
        self.total_steps = 0
        self.models = models
        self.data = data
        self.method = method
        self.optimizers = optimizers
        self.learning_schedules = learning_schedules
        self.eval = eval
        self.reset_models = reset_models
        self.p = p
        if (ema_rates is None):
            self.ema_objects = None
        else:
            #self.ema_models = [ema.EMAHelper(model, mu = mu) for mu in ema_rates]
            logger = eval.logger
            # need to set logger to None for the eval deepcopy
            eval.logger = None
            self.ema_objects = []
            for mu in ema_rates:
                ema_dict = {name: EMAHelper(model, mu = mu) for name, model in self.models.items()}
                ema_dict['eval'] = copy.deepcopy(eval)
                self.ema_objects.append(ema_dict)
            eval.logger = logger 
            for ema_object in self.ema_objects:
                ema_object['eval'].logger = logger
        
        self.kwargs = kwargs
        self.logger = logger
    
    def exists_ls(self, name = 'default'):
        return (name in self.learning_schedules) and (self.learning_schedules[name] is not None)
    
    def train(self, total_epoch, **kwargs):
        tmp_kwargs = copy.deepcopy(self.kwargs)
        tmp_kwargs.update(kwargs)

        def epoch_callback(epoch_loss, models):
            self.eval.register_epoch_loss(epoch_loss)
            self.eval.register_grad_norm(models)
            if self.logger is not None:
                self.logger.log('current_epoch', self.epochs)
        
        def batch_callback(batch_loss):
            # batch_loss is nan, reintialize models
            if np.isnan(batch_loss): 
                print('nan in loss detected, reinitializing models...')
                models, optimizers, learning_schedules = self.reset_models(self.p)
                self.models = models
                self.optimizers = optimizers
                self.learning_schedules = learning_schedules
            
            self.eval.register_batch_loss(batch_loss)
            if self.logger is not None:
                self.logger.log('current_batch', self.total_steps)
            
            # layer_norms = compute_layer_norms(self.models['default'])
            # print('layer norms: min, max, mean, std', torch.min(layer_norms).item(), torch.max(layer_norms).item(), torch.mean(layer_norms).item(), torch.std(layer_norms).item())
        
        self._train_epochs(total_epoch, 
                           epoch_callback=epoch_callback, 
                           batch_callback=batch_callback,
                           **tmp_kwargs)

    def _train_epochs(
                self,
                total_epochs,
                eval_freq = None,
                checkpoint_freq= None,
                checkpoint_callback=None,
                no_ema_eval = False,
                grad_clip = None,
                batch_callback = None,
                epoch_callback = None,
                max_batch_per_epoch = None,
                progress = False,
                refresh_data = False,
                dataset_with_labels = False,
                **kwargs):
        
        for name, model in self.models.items():
            model.train()
        
        print('training model to epoch {} from epoch {}'.format(total_epochs, self.epochs), '...')

        while self.epochs < total_epochs:
            epoch_loss = steps = 0
            for i, Xbatch in enumerate(tqdm(self.data) if progress else self.data):
                if max_batch_per_epoch is not None:
                    if i >= max_batch_per_epoch:
                        break
                
                if dataset_with_labels:
                    Xbatch, y = Xbatch
                    kwargs['y'] = y
                
                training_results = self.method.training_losses(self.models, Xbatch, **kwargs)
                loss = training_results['loss']
                # check if nan in loss
                if torch.isnan(loss):
                    print('nan in loss detected. Saving training_results and breaking loop...')
                    torch.save(training_results, 'nan_training_results.pth')
                    return
                    # print('nan in loss detected, resetting models...')
                    # self.models, self.optimizers, self.learning_schedules = self.reset_models(self.p)
                    # print('models reset.')
                    
                    
                    # print('nan in loss detected, skipping batch...')
                    # for name, model in self.models.items():
                    #     model.zero_grad()
                    continue
                # and finally gradient descent
                for name in self.models:
                    self.optimizers[name].zero_grad()
                loss.backward()
                for name in self.models:
                    if grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.models[name].parameters(), grad_clip)
                    self.optimizers[name].step()
                    if self.exists_ls(name):
                        self.learning_schedules[name].step()
                
                # update ema models
                if self.ema_objects is not None:
                    for e in self.ema_objects:
                        for name in self.models:
                            e[name].update(self.models[name])
                
                epoch_loss += loss.item()
                steps += 1
                self.total_steps += 1
                if batch_callback is not None:
                    batch_callback(loss.item())
            epoch_loss = epoch_loss / steps
            print('epoch_loss', epoch_loss)
            self.epochs += 1
            if epoch_callback is not None:
                epoch_callback(epoch_loss,models=self.models)

            print('Done training epoch {}/{}'.format(self.epochs, total_epochs))
            
            if refresh_data:
                self.data.dataset.refresh_data()

            # now potentially checkpoint
            if (checkpoint_freq is not None) and (self.epochs % checkpoint_freq) == 0:
                checkpoint_callback(self.epochs)
                #print(self.save(curr_epoch=self.manager.training_epochs()))
            
            # now potentially eval
            if (eval_freq is not None) and  (self.epochs % eval_freq) == 0:
                self.evaluate()
                if not no_ema_eval:
                    self.evaluate(evaluate_emas=True)


    def evaluate(self, evaluate_emas = False, **kwargs):
        def ema_callback_on_logging(logger, key, value):
            if not (key in ['losses', 'losses_batch']):
                logger.log('_'.join(('ema', str(ema_obj['default'].mu), str(key))), value)
        
        if not evaluate_emas:
            print('evaluating non-ema model')
            for name, model in self.models.items():
                model.eval()
            with torch.inference_mode():
                self.eval.evaluate_model(self.models, **kwargs)
        elif self.ema_objects is not None:
            for ema_obj in self.ema_objects:
                models = {name: ema_obj[name].get_ema_model() for name in self.models}
                for name in models:
                    models[name].eval()
                with torch.inference_mode():
                    print('evaluating ema model with mu={}'.format(ema_obj['default'].mu))
                    ema_obj['eval'].evaluate_model(models, callback_on_logging = ema_callback_on_logging, **kwargs)
    

    def load(self, filepath):
        # provide key rather than src[key] in case we load an old run that did not contain any vae key
        def safe_load_state_dict(dest, src):
            if dest is not None:
                dest.load_state_dict(src)
        
        checkpoint = torch.load(filepath, map_location=torch.device(self.method.device))
        self.total_steps = checkpoint['steps']
        self.epochs = checkpoint['epoch']
        for name in self.models:
            chckpt_model_name = 'model_{}_parameters'.format(name) if name != 'default' else 'model_parameters' # retro-compatibility: 'default' becomes ''
            chckpt_optim_name = 'optimizer_{}'.format(name) if name != 'default' else 'optimizer'
            chckpt_ls_name = 'learnin_schedule_{}'.format(name) if name != 'default' else 'learning_schedule'
            chckpt_ema_name = 'ema_models_{}'.format(name) if name != 'default' else 'ema_models'
            print('loading model {} from checkpoint'.format(name))
            safe_load_state_dict(self.models[name], checkpoint[chckpt_model_name])
            print('loading optimizer {} from checkpoint'.format(name))
            safe_load_state_dict(self.optimizers[name], checkpoint[chckpt_optim_name])
            print('loading learning schedule {} from checkpoint'.format(name))
            safe_load_state_dict(self.learning_schedules[name], checkpoint[chckpt_ls_name])
            if self.ema_objects is not None:
                assert chckpt_ema_name in checkpoint, 'no ema model in checkpoint'
                for ema_obj, ema_state in zip(self.ema_objects, checkpoint[chckpt_ema_name]):
                    print('loading ema model {} with mu={} from checkpoint'.format(name, ema_obj[name].mu))
                    safe_load_state_dict(ema_obj[name], ema_state)

            
    def save(self, filepath):
        def safe_save_state_dict(src):
            return src.state_dict() if src is not None else None
        checkpoint = {
            'epoch': self.epochs,
            'steps': self.total_steps,
        }
        for name in self.models:
            chckpt_model_name = 'model_{}_parameters'.format(name) if name != 'default' else 'model_parameters' # retro-compatibility: 'default' becomes ''
            chckpt_optim_name = 'optimizer_{}'.format(name) if name != 'default' else 'optimizer'
            chckpt_ls_name = 'learnin_schedule_{}'.format(name) if name != 'default' else 'learning_schedule'
            chckpt_ema_name = 'ema_models_{}'.format(name) if name != 'default' else 'ema_models'
            checkpoint[chckpt_model_name] = safe_save_state_dict(self.models[name])
            checkpoint[chckpt_optim_name] = safe_save_state_dict(self.optimizers[name])
            checkpoint[chckpt_ls_name] = safe_save_state_dict(self.learning_schedules[name])
            if self.ema_objects is not None:
                checkpoint[chckpt_ema_name] = [safe_save_state_dict(ema_obj[name]) for ema_obj in self.ema_objects]

        torch.save(checkpoint, filepath)
    
    def save_eval_metrics(self, eval_path):
        eval_save = {'eval': self.eval.evals}
        if self.ema_objects is not None:
            eval_save.update({'ema_evals': [(ema_obj['eval'].evals, ema_obj['default'].mu) for ema_obj in self.ema_objects]})
        torch.save(eval_save, eval_path)
    
    def load_eval_metrics(self, eval_path):
        eval_save = torch.load(eval_path)
        assert 'eval' in eval_save, 'no eval subdict in eval file'
        # load eval metrics
        self.eval.evals.update(eval_save['eval'])
        self.eval.log_existing_eval_values(folder='eval')

        # load ema eval metrics
        if not 'ema_evals' in eval_save:
            return
        assert self.ema_objects is not None
        # saved ema evaluation, in order
        saved_ema_evals = [ema_eval_save for ema_eval_save, mu_save in eval_save['ema_evals']]
        # saved ema mu , in order
        saved_mus = [mu_save for ema_eval_save, mu_save in eval_save['ema_evals']]
        
        for ema_obj in self.ema_objects:
            # if mu has not been run previously, no loading
            if ema_obj['default'].mu not in saved_mus:
                continue
            # find index of our mu of interest
            idx = saved_mus.index(ema_obj['default'].mu)
            # load the saved evaluation
            ema_obj['eval'].evals.update(saved_ema_evals[idx])
            # log the saved evaluation
            ema_obj['eval'].log_existing_eval_values(folder='eval_ema_{}'.format(ema_obj['default'].mu))
    