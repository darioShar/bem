import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import os 

from .fid_score import fid_score, prdc
import torchvision.utils as tvu
from .wasserstein import compute_wasserstein_distance, compute_sliced_wasserstein
from .prd_legacy import compute_precision_recall_curve, compute_f_beta
from .mmd_loss import MMD_loss
from .discrete_losses import compute_kl_div_and_hellinger
from .msle import get_msle_empirical

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def generate_and_save_data(gen_manager, 
                           gen_data_path,
                           data_to_generate,
                           models, 
                           batch_size,
                            **kwargs
                           ):
    # generate more data if necessary
    #data_to_gen_wass : really is the batch size now. We use some approximation.
    remaining = data_to_generate
    print('generating {} images for fid computation'.format(remaining))
    total_generated_data = 0
    while remaining > 0:
        print(remaining, end = ' ')
        gen_manager.generate(models,
                                min(batch_size, remaining),
                                **kwargs)
        # save data to file. We do that rather than concatenating to save on memory, 
        # but really it is because I want to inspect the images while they are generated
        print('saving {} generated images'.format(gen_manager.samples.shape[0]))
        for i in range(gen_manager.samples.shape[0]):
            tvu.save_image(gen_manager.samples[i].float(), os.path.join(gen_data_path, f"{i+total_generated_data}.png"))
        total_generated_data += gen_manager.samples.shape[0]
        #gen_samples = torch.cat((gen_samples, gen_model.samples), dim = 0)
        remaining = remaining - gen_manager.samples.shape[0]
        
    assert data_to_generate == total_generated_data
    print('saved generated data in {}.'.format(gen_data_path))



class EvaluationManager:

    def __init__(self,
                 method,
                 gen_manager,
                 dataloader,
                 verbose = True,
                 logger = None,
                 modality = None,
                 state_space = None,
                 gen_data_path = None,
                 real_data_path = None,
                 **kwargs):

        self.method = method
        self.gen_manager = gen_manager
        self.dataloader = dataloader
        self.verbose = verbose
        self.logger = logger
        self.modality = modality
        self.state_space = state_space
        self.gen_data_path = gen_data_path
        self.real_data_path = real_data_path
        self.kwargs = kwargs
        self.reset()
        '''self.gen_model = gen_model Gen.GenerationManager(model = self.model,
                                               method = self.method,
                                              dataloader=self.dataloader,
                                              is_image = is_image)'''



    def reset(self,
              keep_losses = False,
              keep_evals = False):
        self.evals = {
            'losses': self.evals['losses'] if keep_losses else np.array([], dtype = np.float32),
            'losses_batch': self.evals['losses_batch'] if keep_losses else np.array([], dtype = np.float32),
            'grad_norm': self.evals['grad_norm'] if keep_evals else np.array([], dtype = np.float32),
            
            # small dimensional data
            # continuous
            'wass': self.evals['wass'] if keep_evals else [],
            'mmd': self.evals['mmd'] if keep_evals else [],
            'msle': self.evals['msle'] if keep_evals else [],
            # discrete
            'hellinger': self.evals['hellinger'] if keep_evals else [],
            'kl_div': self.evals['kl_div'] if keep_evals else [],
            'sliced_wass': self.evals['sliced_wass'] if keep_evals else [],
            
            # image data
            'precision': self.evals['precision'] if keep_evals else [],
            'recall': self.evals['recall'] if keep_evals else [],
            'f_1_pr': self.evals['f_1_pr'] if keep_evals else [],
            'density': self.evals['density'] if keep_evals else [],
            'coverage': self.evals['coverage'] if keep_evals else [],
            'f_1_dc': self.evals['f_1_dc'] if keep_evals else [],
            'fid': self.evals['fid'] if keep_evals else [],
            
            'fig': self.evals['fig'] if keep_evals else [],

        }
    
    #def save(self, eval_path):
    #    torch.save(self.evals, eval_path)

    def log_existing_eval_values(self, folder='eval'):
        #self.evals = torch.load(eval_path)
        if self.logger is not None:
            new_values = {folder: self.evals}
            self.logger.set_values(new_values)
    
    def register_batch_loss(self, batch_loss):
        self.evals['losses_batch'] = np.append(self.evals['losses_batch'], batch_loss)
        if self.logger is not None:
            self.logger.log('losses_batch', batch_loss)
    
    def register_epoch_loss(self, epoch_loss):
        self.evals['losses'] = np.append(self.evals['losses'], epoch_loss)
        if self.logger is not None:
            self.logger.log('losses', self.evals['losses'][-1])

    def register_grad_norm(self, models):
        self.evals['grad_norm'] = np.append(self.evals['grad_norm'], compute_gradient_norm(models['default']))
        if self.logger is not None:
            self.logger.log('grad_norm', self.evals['grad_norm'][-1])

    # uses default parameters for generation. Return generation manager
    def generate_default(self, models, nsamples, **kwargs):
        self.gen_manager.generate(models, nsamples, **kwargs)
        return self.gen_manager

    # little workaround to enable arbitrary number of kwargs to be specified beforehand
    def evaluate_model(self, models, **kwargs):
        tmp_kwargs = copy.deepcopy(self.kwargs)
        tmp_kwargs.update(kwargs)
        self._evaluate_model(models, **tmp_kwargs)

    # compute evaluatin metrics
    def _evaluate_model(self,
                        models,
                        data_to_generate,
                        batch_size,
                        fig_lim = 1.5,
                        callback_on_logging = None,
                        **kwargs):
        
        eval_results = {}
        
        is_image = self.modality == 'image'
        is_continuous = self.state_space == 'continuous'
        
        print('modality = {}, state_space = {}, {} samples to generate for evaluation'.format(self.modality, self.state_space, data_to_generate))
        print('computing metrics...')
        if not is_image:
            
            # data_to_generate #min(data_to_generate, 128)
            self.gen_manager.generate(models, data_to_generate, **kwargs)
            
            # prepare data.
            gen_samples = self.gen_manager.samples.cpu()
            
            data = self.gen_manager.load_original_data(data_to_generate)
            
            if is_continuous:
                print('wasserstein')
                eval_results['wass'] = compute_wasserstein_distance(data, 
                                                        gen_samples, 
                                                        bins = 250 if data_to_generate >=512 else 'auto')
                print('mmd')
                eval_results['mmd'] = MMD_loss()(data.squeeze(1), gen_samples.squeeze(1))#, kernel='rbf')#get_MMD(data, gen_samples, data[0].device)
                
                print('msle')
                eval_results['msle'] = get_msle_empirical(data, gen_samples, agg = 'mean')
                
                print('prd')
                pr_curve = compute_precision_recall_curve(data, 
                                                        gen_samples, 
                                                        num_clusters=100 if data_to_generate > 2500 else 20)
                f_beta = compute_f_beta(*pr_curve)
                prdc_value = {
                    'precision': f_beta[0],
                    'recall': f_beta[1],
                }
                # compute f_1 scores
                prec, rec = prdc_value['precision'], prdc_value['recall']
                eval_results['f_1_pr'] = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.
                
            else:
                # if the data dimension gets bigger, we only compute sliced wassersetin metric
                data_dim = torch.prod(torch.tensor(data.shape[1:])).item()
                only_compute_sliced_wasserstein = data_dim > 8
                print('data dimension is {}, only compute sliced wass (d >= 8) = {}'.format(data_dim, only_compute_sliced_wasserstein))
                
                if only_compute_sliced_wasserstein:
                    # fill with zero for retro-compatibility
                    eval_results['kl_div'] = 0.
                    eval_results['hellinger'] = 0.
                else:
                    print('kl_div and hellinger')
                    kl_div, hellinger = compute_kl_div_and_hellinger(data, gen_samples)
                    eval_results['kl_div'] = kl_div
                    eval_results['hellinger'] = hellinger
                
                # sliced wasserstein
                print('sliced_wass')
                sliced_wass = compute_sliced_wasserstein(data, gen_samples, n_projections=1000)
                eval_results['sliced_wass'] = sliced_wass
            
        else:
            if data_to_generate != 0:
                generate_and_save_data(self.gen_manager,
                                        self.gen_data_path,
                                        data_to_generate,
                                        models,
                                        batch_size,
                                        **kwargs)
            
            # fid score
            print('fid')
            eval_results['fid'] = fid_score(self.real_data_path, 
                                            self.gen_data_path, 
                                            batch_size, # batch size
                                            self.method.device, 
                                            num_workers= 2 if is_image else 0)            
            
            print('prdc')
            # precision, recall density, coverage
            prdc_value = prdc(self.real_data_path, 
                            self.gen_data_path, 
                            batch_size, # batch size 
                            self.method.device, 
                            num_workers= 2 if is_image else 0,
                            max_num_files=data_to_generate if data_to_generate != 0 else None) # None means read whole image directory, which should be full from a previous run
        
            for k, v in prdc_value.items():
                eval_results[k] = v
            
            # compute f_1 scores
            prec, rec = prdc_value['precision'], prdc_value['recall']
            eval_results['f_1_pr'] = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.
            den, cov = prdc_value['density'], prdc_value['coverage']
            eval_results['f_1_dc'] = (2 * den * cov) / (den + cov) if den + cov > 0 else 0.
            
            
        # TODO: figure
        # if not is_image:
        #     fig = self.gen_manager.get_plot(xlim = (-fig_lim, fig_lim), ylim = (-fig_lim, fig_lim))
        # else:
        #     if len(self.gen_manager.samples) != 0:
        #         fig = self.gen_manager.get_image() # todo: load an image from the folder
        #     else:
        #         fig = None
        # eval_results['fig'] = fig
        # plt.close(fig)

        if self.logger is not None:
            for k, v in eval_results.items():
                if callback_on_logging is not None:
                    callback_on_logging(self.logger, k, v)
                else:
                    self.logger.log(k, v)
        
        # for the moment, do not save the figures. Compatibility issues between different versions of matplotlib,
        # and thus for different systems. 
        eval_results['fig'] = None

        # append results to self.evals dictionnary
        for k in eval_results.keys():
            self.evals[k].append(eval_results[k])

        # print them if necessary
        if self.verbose:
            print('results:')
            # last loss, if computed
            if self.evals['losses_batch'].any():
                print(f"losses_batch = {self.evals['losses_batch'][-1]}")
            # and the valuation metrics
            for k, v in eval_results.items():
                print('{} = {}'.format(k, v))
        
        return eval_results

