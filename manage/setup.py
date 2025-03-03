import torch
import numpy as np




'''
        check for device on which to run Pytorch
'''
def _get_device():
    if torch.backends.mps.is_available():
        device = "mps"
        mps_device = torch.device(device)
    elif torch.cuda.is_available():
        device = "cuda"
        cuda_device = torch.device(device)
    else:
        device = 'cpu'
        print ("GPU device not found.")
    print ('using device {}'.format(device))
    return device


'''
    If running with cuda, perform some optimizations
'''
def _optimize_gpu(device):
    if device == 'cuda':
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.backends.cudnn.benchmark = True

'''
    Set random seeds for torch, numpy, and cuda torch
'''
def _set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


