import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, CelebA, MNIST, ImageNet, ImageFolder
from datasets.tinyimagenet import TinyImageNetDataset
from torch.utils.data import TensorDataset
from datasets.lsun import LSUN
import torch.utils.data
from torch.utils.data import Subset, Dataset
import pickle
from PIL import Image
from torch.utils.data import DataLoader

from datasets.custom_distributions import CustomDistributionDataset
from datasets.png_to_2d_dataset import get_dataset_from_png, check_png_dataset_exists
from datasets.img_datasets import get_img_dataset


class Modality:
    IMAGE = 'image'
    TENSOR_RANK_TWO = 'tensor_rank_two'

class StateSpace:
    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'

# this is a gloabl class to check the dataset type
class CurrentDatasetInfo:
    # static attributes
    modality = Modality.IMAGE
    state_space = StateSpace.CONTINUOUS
    has_labels = False
    
    @staticmethod
    def set_dataset_info(modality, state_space, has_labels):
        
        # assert modality is in the attribute list of Modality class
        assert modality in [Modality.IMAGE, Modality.TENSOR_RANK_TWO], \
                "received {}, but modality must be 'image' or 'tensor_rank_two' (Channelsxdimension dataset) ".format(modality)
        assert state_space in [StateSpace.CONTINUOUS, StateSpace.DISCRETE], \
            "received {}, but state_space must be 'continuous' or 'discrete'".format(state_space)
        assert isinstance(has_labels, bool), 'has_labels must be a boolean'
        
        CurrentDatasetInfo.modality = modality
        CurrentDatasetInfo.state_space = state_space
        CurrentDatasetInfo.has_labels = has_labels
        print('Dataset type set to: {}, has_labels={}'.format(modality, state_space, has_labels))



DATA_PATH = './data'


def available_datasets():
    return {
        'image': ['cifar10', 'cifar10_lt', 'celeba', 'mnist', 'binary_mnist', 'imagenet', 'tinyimagenet'],
        'custom': list(CustomDistributionDataset.available_distributions.keys()),
        'png': 'See available png images in the ./data folder'
    }

def get_dataset(p):
    
    dataset = p['data']['dataset'].lower()
    print('loading dataset {}'.format(dataset))
    
    available = available_datasets()
    
    if dataset in available['custom']:
        train_dataset = CustomDistributionDataset(**p['data'])
        train_dataset.generate()
        test_dataset = CustomDistributionDataset(**p['data'])
        test_dataset.generate()
        
        modality = 'tensor_rank_two'
        state_space = train_dataset.state_space
        has_labels = False # can set an entry in p to determine if we use labels or not in the dataset
        
        return train_dataset, test_dataset, modality, state_space, has_labels
    
    if dataset in available['image']:
        train_dataset, test_dataset, is_discrete = get_img_dataset(
            DATA_PATH,
            dataset, 
            p['data']['image_size'],
            p['data']['random_flip'],
            lsun_category=None
        )
        
        modality='tensor_rank_two'
        state_space='discrete' if is_discrete else 'continuous'
        has_labels = True
        
        return train_dataset, test_dataset, modality, state_space, has_labels
    
    elif check_png_dataset_exists(DATA_PATH, dataset): 
        # check if dataset is available as simple png image
        nsamples = p['data']['nsamples']
        train_dataset = get_dataset_from_png(DATA_PATH, dataset, nsamples)
        test_dataset = get_dataset_from_png(DATA_PATH, dataset, nsamples)
    
        modality='tensor_rank_two'
        state_space='continuous'
        has_labels = False
        
        return train_dataset, test_dataset, modality, state_space, has_labels
    
    else:
        raise ValueError('Dataset {} not found. Available datasets are: {}'.format(dataset, available_datasets()))
        