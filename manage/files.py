import torch
from pathlib import Path
import yaml
import os
import hashlib
from torchvision import utils as tvu


''''''''''' FILE MANIPULATION '''''''''''
class FileHandler:
    '''
    p: parameters dictionnary
    '''
    def __init__(self, 
                 exp_name = None, 
                 eval_name = None):
        self.exp_name = exp_name if exp_name is not None else FileHandler.default_exp_name
        self.eval_name = eval_name if eval_name is not None else FileHandler.default_eval_name
    

    @staticmethod
    def default_exp_name(p, verbose=False, limit_str_len = 5):
        model_param = p['model']
        default_exp_dict = {
            # 'data': {k:v for k, v in p['data'].items() if k in ['dataset', 'channels', 'image_size']},
            'data': {k:v for k, v in p['data'].items() if k in ['dataset', 'image_size']},
            # p['method']: {k:v for k, v in p[p['method']].items()},
            'method': p['method'],
            'model':  {k:v for k, v in model_param.items() if k in ['architecture']}, 
        }
        
        # tmp = 'exp_{}_{}_{}'.format(p['method'], p['data']['dataset'], datetime.now().strftime('%d_%m_%y_%H_%M_%S'))
        
        L = []
        for k, v in default_exp_dict.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    L.append(k1)
                    L.append(v1)
            else:
                L.append(k)
                L.append(v)
        L = [str(x)[:limit_str_len] for x in L]
        res = '_'.join(L)
        # res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
        # res = str(res)[:16]
        #res = str(hex(abs(hash(tuple(p)))))[2:]
        if verbose:
            print('experiment name: {} \n\tParameters: {}'.format(res, default_exp_dict))
        return res

    
    @staticmethod
    def default_eval_name(p, verbose = False, limit_str_len = 5):
        default_eval_dict =  {'eval': p['eval'][p['method']]}
        # res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
        # res = str(res)[:8]
        L = []
        for k, v in default_eval_dict.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    L.append(k1)
                    L.append(v1)
            else:
                L.append(k)
                L.append(v)
        L = [str(x)[:limit_str_len] for x in L]
        res = '_'.join(L)
        if verbose:
            print('eval name: {} \n\tParameters: {}'.format(res, default_eval_dict))
        return res

    # returns new save folder and parameters hash
    def get_exp_path_from_param(self,
                                 p,
                                folder_path, 
                                make_new_dir = False):
        
        h = self.exp_name(p)
        save_folder_path = os.path.join(folder_path, p['data']['dataset'])
        # limit the length of the path to 255 characters
        save_folder_path = save_folder_path[:255]
        if make_new_dir:
            Path(save_folder_path).mkdir(parents=True, exist_ok=True)
        return save_folder_path, h

    # returns eval folder given save folder, and eval hash
    def get_eval_path_from_param(self, 
                                p,
                                save_folder_path, 
                                make_new_dir = False):
        h = self.exp_name(p)
        h_eval = self.eval_name(p)
        eval_folder_path = os.path.join(save_folder_path, '_'.join(('new_eval', h, h_eval)))
        # limit the length of the path to 255 characters
        eval_folder_path = eval_folder_path[:255]
        if make_new_dir:
            Path(eval_folder_path).mkdir(parents=True, exist_ok=True)
        return eval_folder_path, h_eval

    # returns paths for model and param
    # from a base folder. base/data_distribution/
    def get_paths_from_param(self, 
                             p,
                            folder_path, 
                            make_new_dir = False, 
                            curr_epoch = None, 
                            new_eval_subdir=False,
                            do_not_load_model=False
                            ): # saves eval and param in a new subfolder
        save_folder_path, h = self.get_exp_path_from_param(p, folder_path, make_new_dir)
        if new_eval_subdir:
            eval_folder_path, h_eval = self.get_eval_path_from_param(p, save_folder_path, make_new_dir)

        names = ['model', 'parameters', 'eval']
        # create path for each name
        # in any case, model get saved in save_folder_path
        if curr_epoch is not None:
            L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(curr_epoch)])}
        else:
            L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
            if not do_not_load_model:
                # checks if model is there. otherwise, loads latest model. also checks equality of no_iteration model and latest iteration one
                # list all model iterations
                model_paths = list(Path(save_folder_path).glob('_'.join(('model', h)) + '*'))
                assert len(model_paths) > 0, 'no models to load in {}, with hash {}'.format(save_folder_path, h)
                max_model_iteration = 0
                max_model_iteration_path = None
                for i, x in enumerate(model_paths):
                    if str(x)[:-3].split('_')[-1].isdigit() and (len(str(x)[:-3].split('_')[-1]) < 8): # if it is digit, and not hash
                        model_iter = int(str(x)[:-3].split('_')[-1])
                        if max_model_iteration< model_iter:
                            max_model_iteration = model_iter
                            max_model_iteration_path = str(x)
                if max_model_iteration_path is not None:
                    if Path(L['model'] + '.pt').exists():
                        print('Found another save with no specified iteration alonside others with specified iterations. Will not load it')
                    print('Loading trained model at iteration {}'.format(max_model_iteration))
                    L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(max_model_iteration)])}
                elif Path(L['model']+ '.pt').exists():
                    print('Found model with no specified iteration. Loading it')
                    # L already holds the right name
                    #L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
                else:
                    raise Exception('Did not find a model to load at location {} with hash {}'.format(save_folder_path, h))
                
        # then depending on save_new_eval, save either in save_folder or eval_folder
        if new_eval_subdir:
            if curr_epoch is not None:
                L.update({name: '_'.join([os.path.join(eval_folder_path, name), h_eval, str(curr_epoch)]) for name in names[1:]})
            else:
                L.update({name: '_'.join([os.path.join(eval_folder_path, name), h_eval]) for name in names[1:]})
        else:
            # we consider the evaluation to be made all along the epochs, in order to get a list of evaluations.s
            # so we do not append curr_epoch here. 
            L.update({name: '_'.join([os.path.join(save_folder_path, name), h]) for name in names[1:]})
        
        return tuple(L[name] +'.pt' for name in L.keys()) # model, param, eval

    def _prepare_data_directories(self, 
                                  dataset_name, 
                                  dataset_files, 
                                  remove_existing_eval_files, 
                                  num_real_data, 
                                  hash_params,
                                  is_image_dataset):

        if dataset_files is None:
            # do nothing, assume no data will be generated
            print('(prepare data directories) assuming no data will be generated.')
            return None, None

        # create directory for saving images
        folder_path = os.path.join('eval_files', dataset_name)
        generated_data_path = os.path.join(folder_path, 'generated_data', hash_params)
        if not is_image_dataset:
            # then we have various versions of the same dataset
            real_data_path = os.path.join(folder_path, 'original_data', hash_params)
        else:
            real_data_path = os.path.join(folder_path, 'original_data')
        
        #Path(generated_data_path).mkdir(parents=True, exist_ok=True)
        #Path(real_data_path).mkdir(parents=True, exist_ok=True)

        def remove_file_from_directory(dir):
            # remove the directory
            if not dir.is_dir():
                raise ValueError(f'{dir} is not a directory')
            # print('removing files in directory', dir)
            for file in dir.iterdir():
                file.unlink()

        def save_images(path):
                print('storing dataset in', path)
                # now saving the original data
                # assert dataset_name.lower() in ['mnist', 'cifar10', 'celeba', 'cifar10_lt', 'tinyimagenet'], 'only mnist, cifar10, celeba datasets are supported for the moment. \
                #     For the moment we are loading {} data points. We may need more for the other datasets, \
                #         and anyway we should implement somehting more systematic'.format(num_real_data)
                #data = gen_model.load_original_data(evaluation_files) # load all the data. Number of datapoints specific to mnist and cifar10
                data_to_store = num_real_data
                print('saving {} original images from pool of {} datapoints'.format(data_to_store, len(dataset_files)))
                for i in range(data_to_store):
                    if (i%500) == 0:
                        print(i, end=' ')
                    tvu.save_image(dataset_files[i][0]), os.path.join(path, f"{i}.png")
        
        path = Path(generated_data_path)
        if path.exists():
            if remove_existing_eval_files:
                remove_file_from_directory(path)
        else:
            path.mkdir(parents=True, exist_ok=True)

        path = Path(real_data_path)
        if is_image_dataset:
            if path.exists():
                print('found', path)
                assert path.is_dir(), (f'{path} is not a directory')
                # check that there are the right number of image files, else remove and regenerate
                if len(list(path.iterdir())) != num_real_data:
                    remove_file_from_directory(path)
                    save_images(path)
            else:
                path.mkdir(parents=True, exist_ok=True)
                save_images(path)
        else:
            if path.exists():
                remove_file_from_directory(path)
            else:
                path.mkdir(parents=True, exist_ok=True)

        return generated_data_path, real_data_path

    def prepare_data_directories(self, p, dataset_files, is_image_dataset):
        # prepare the evaluation directories
        return self._prepare_data_directories(dataset_name=p['data']['dataset'],
                                dataset_files = dataset_files, 
                                remove_existing_eval_files = False if p['eval']['data_to_generate'] == 0 else True,
                                num_real_data = p['eval']['real_data'],
                                hash_params = '_'.join([self.exp_name(p), self.eval_name(p)]), # for saving images. We want a hash specific to the training, and to the sampling
                                is_image_dataset = is_image_dataset
                                )
    @staticmethod
    def get_param_from_config(config_path, config_file):
        with open(os.path.join(config_path, config_file), "r") as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    # loads all params from a specific folder
    def get_params_from_folder(folder_path):
        return [(path, torch.load(path)) for path in Path(folder_path).glob("parameters*")]





        #save_folder_path, h = get_hash_path_from_param(p, folder_path, make_new_dir)
        #if new_eval_subdir:
        #    eval_folder_path, h, h_eval = get_eval_path_from_param(p, save_folder_path, make_new_dir)
    #
        #names = ['model', 'parameters', 'eval']
        ## create path for each name
        ## in any case, model get saved in save_folder_path
        #if curr_epoch is not None:
        #    L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(curr_epoch)])}
        #else:
        #    L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
        ## then depending on save_new_eval, save either in save_folder or eval_folder
        #if new_eval_subdir:
        #    if curr_epoch is not None:
        #        L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval, str(curr_epoch)]) for name in names[1:]})
        #    else:
        #        L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval]) for name in names[1:]})
        #else:
        #    # we consider the evaluation to be made all along the epochs, in order to get a list of evaluations.s
        #    # so we do not append curr_epoch here. 
        #    L.update({name: '_'.join([os.path.join(save_folder_path, name), h]) for name in names[1:]})
        #
        #return tuple(L[name] +'.pt' for name in L.keys()) # model, param, eval
