# file to generate some default data like swiss roll, GMM, levy variables

from sklearn.datasets import make_swiss_roll
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import scipy
import numpy as np
from inspect import signature
from torch.utils.data import Dataset
from .torchlevy.levy import LevyStable

def match_last_dims(data, size):
    # Ensure data is 1-dimensional
    assert data.dim() == 1, f"Data must be 1-dimensional, got {data.size()}"

    # Unsqueeze to add singleton dimensions for expansion
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    
    # Use expand instead of repeat to save memory
    return data.expand(*size)



''' Generate fat tail distributions'''
# assumes it is a batch size
# is isotropic, just generates a single 'a' tensored to the right shape
def gen_skewed_levy(alpha, 
                    size, 
                    device = None, 
                    isotropic = True,
                    clamp_a = None):
    if (alpha > 2.0 or alpha <= 0.):
        raise Exception('Wrong value of alpha ({}) for skewed levy r.v generation'.format(alpha))
    if alpha == 2.0:
        ret = 2 * torch.ones(size)
        return ret if device is None else ret.to(device)
    # generates the alplha/2, 1, 0, 2*np.cos(np.pi*alpha/4)**(2/alpha)
    if isotropic:
        ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0, scale=2*np.cos(np.pi*alpha/4)**(2/alpha), size=size[0]), dtype=torch.float32)
        ret = match_last_dims(ret, size)
    else:
        ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0, scale=2*np.cos(np.pi*alpha/4)**(2/alpha), size=size), dtype=torch.float32)
    if clamp_a is not None:
        ret = torch.clamp(ret, 0., clamp_a)
    return ret if device is None else ret.to(device)


#symmetric alpha stable noise of scale 1 can generate from totally skewed noise if provided assumes it is a batch size
def gen_sas(alpha, 
            size, 
            a = None, 
            device = None, 
            isotropic = True,
            clamp_eps = None):
    if a is None:
        a = gen_skewed_levy(alpha, size, device = device, isotropic = isotropic)
    ret = torch.randn(size=size, device=device)
    ret = torch.sqrt(a)* ret
    if clamp_eps is not None:
        ret = torch.clamp(ret, -clamp_eps, clamp_eps)
    return ret


def _between_minus_1_1_with_quantile(x, quantile, scale_to_minus_1_1 = True):
    # assume x is centred
    high_quantile = torch.quantile(x, quantile, dim = 0, interpolation='nearest')
    low_quantile = torch.quantile(x, 1 - quantile, dim = 0, interpolation='nearest')
    assert not (high_quantile < 0).any()
    assert not (low_quantile > 0).any()
    clamp_value = torch.max(torch.abs(high_quantile), torch.abs(low_quantile))
    clamp_value = clamp_value.unsqueeze(0).repeat(x.shape[0], *tuple([1]* len(x.shape[1:])))
    tmp = torch.clamp(x, min= - clamp_value, max=clamp_value)
    idx_high = (tmp >= clamp_value)
    idx_low = (tmp <= - clamp_value)
    tmp[idx_high] = clamp_value[idx_high]
    tmp[idx_low] = -clamp_value[idx_low]
    if scale_to_minus_1_1:
        tmp /= clamp_value
    else:
        # just clamp
        tmp = tmp.clamp(-1, 1)
        #tmp = tmp
    return tmp



def sample_2_gmm(n_samples, 
                 alpha = None, 
                 n = None, 
                 std = None, 
                 theta = 1.0, 
                 weights = None, 
                 device = None, 
                 normalize=False, 
                 isotropic = False,
                 between_minus_1_1 = False,
                    quantile_cutoff = 1.0):
    if weights is None:
        weights = np.array([0.5, 0.5])
    means = np.array([ [theta, 0], [-theta, 0] ])
    gmm = GaussianMixture(n_components=2)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(2)]
    x, _ = gmm.sample(n_samples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff, scale_to_minus_1_1=True) # should do something with 1 / sqrt(n)
    x[torch.randperm(x.size()[0])] # shuffle rows
    x = x.unsqueeze(1) # add channel dimension
    return x

def sample_grid_gmm(n_samples, 
                    alpha = None, 
                    d = 2,
                    n = None, 
                    std = None, 
                    theta = None, 
                    weights = None, 
                    device = None, 
                    normalize=False, 
                    isotropic = False,
                    between_minus_1_1 = False,
                    quantile_cutoff = 1.0):
    if weights is None:
        weights = np.array([1 / (n*n) for i in range(n*n)])
    means = []
    for i in range(n):
        for j in range(n):
            means.append([2*i/(n - 1) - 1, 2*j/(n-1) - 1])
    means = np.array(means)
    gmm = GaussianMixture(n_components=n*n)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(n*n)]
    x, _ = gmm.sample(n_samples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff) # should do something with 1 / sqrt(n)
    
    x[torch.randperm(x.size()[0])] # shuffle rows
    x = x.unsqueeze(1) # add channel dimension
    return x[torch.randperm(x.size()[0])]


def gen_swiss_roll(n_samples, 
                   alpha = None, 
                   n = None, 
                   std = None, 
                   theta = None, 
                   weights = None, 
                   device = None, 
                   normalize=False, 
                   isotropic = False,
                   between_minus_1_1 = False,
                    quantile_cutoff = 1.0):
    x, _ = make_swiss_roll(n_samples=n_samples, noise=std)
    # Make two-dimensional to easen visualization
    x = x[:, [0, 2]]
    x = (x - x.mean()) / x.std()
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff) # should do something with 1 / sqrt(n)
    
    x[torch.randperm(x.size()[0])] # shuffle rows
    x = x.unsqueeze(1) # add channel dimension
    return torch.tensor(x, dtype = torch.float32)


def sample_grid_sas(n_samples,
                    alpha = 1.8, 
                    d = 2,
                    n = None, 
                    std = None, 
                    theta = 1.0, 
                    weights = None, 
                    device = None, 
                    normalize=False, 
                    isotropic = False,
                    between_minus_1_1 = False,
                    quantile_cutoff = 1.0):
    assert d <= 2, 'Only 2D grid is supported for now'
    if weights is None:
        weights = np.array([1 / (n*n) for i in range(n*n)])
    data = std * gen_sas(alpha, size = (n_samples, 2), isotropic = isotropic)
    weights = np.concatenate((np.array([0.0]), weights))
    idx = np.cumsum(weights)*n_samples
    for i in range(n):
        for j in range(n):
            # for the moment just selecting exact proportions
            s = int(idx[i*n + j])
            e = int(idx[i*n + j + 1])
            data[s:e] = data[s:e] + torch.tensor([2*i/(n - 1) - 1, 2*j/(n-1) - 1])
    

    if normalize:
        data = (data - data.mean()) / data.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    #data = torch.tensor(data, dtype = torch.float32)
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff) # should do something with 1 / sqrt(n)
    
    if d == 1:
        data = data[:, 0].unsqueeze(1)
    x = data
    x[torch.randperm(x.size()[0])] # shuffle rows
    x = x.unsqueeze(1) # add channel dimension
    return x

# give a function 'f' defined in [d] with output in (0, 1)^d. Defines 
# the independent probability of observing each components, such that X_i ~ Bernoulli(f_i)
def sample_discrete_hypercube(val: torch.Tensor) -> torch.Tensor:
    # Generate uniform random in [0,1], same shape as val
    rnd = torch.rand_like(val)
    # If rnd < val, then 1, else 0
    return (rnd < val).float()

def saws_1d(n_samples, d, num_saws = 2, shift=0.):
    def saw(shift):
        # start at shift, oscillate num_saws times
        x = torch.arange(0, d) / d
        rescaled_val = (x - shift) * (2*num_saws)
        anterior_int = np.floor(rescaled_val)
        posterior_int = anterior_int + 1
        anterior_value = (anterior_int % 2)
        posterior_value = (anterior_value +1) % 2
        # rescale [0,1] to [1-rescaling, rescaling]
        rescaling = 0.9
        anterior_value = ((anterior_value*2 - 1)*rescaling + 1) / 2
        posterior_value = ((posterior_value*2 - 1)*rescaling +1)/2
        val = anterior_value + (rescaled_val - anterior_int)*(posterior_value - anterior_value)
        return val
    if (type(shift) == float) or (type(shift) == int):
        return saw(shift).unsqueeze(0).unsqueeze(0).repeat(n_samples, 1, 1)
    else:
        return torch.stack([saw(shift[i]) for i in range(n_samples)]).unsqueeze(1)

def sample_saws_1d(n_samples, d, num_saws = 2, shift=0.):
    return sample_discrete_hypercube(saws_1d(n_samples, d, num_saws=num_saws, shift=shift))

def sample_uniform_saws_1d(n_samples, d, num_saws = 2):
    return sample_discrete_hypercube(saws_1d(n_samples, d, num_saws=num_saws, shift=np.random.rand(n_samples)))

# we can have a more complicated function, taking for instance X ~ Normal(0, I_n)
# as input, 
# and outputting the bernoulli parameter for each component. We can train
# VAE on binary MNIST.





# Wrapper class to create a Dataset from the simple distributions defined in .Distributions.py
# We can specify arbitrary *args and **kwargs. They must match with the selected function signature
class CustomDistributionDataset(Dataset):

    available_distributions = \
        {'gmm_2': sample_2_gmm,
        'gmm_grid': sample_grid_gmm,
        'swiss_roll':gen_swiss_roll,
        'skewed_levy': gen_skewed_levy, # self.levy_stable.gen_skewed_levy,#gen_skewed_levy,
        'sas': gen_sas,#self.levy_stable.gen_sas,#gen_sas,
        #'gaussian_noising': gaussian_noising,
        #'stable_noising': stable_noising,
        'sas_grid': sample_grid_sas,
        'saws_1d': sample_saws_1d,
        'uniform_saws_1d':sample_uniform_saws_1d
        }
    
    def __init__(self, dataset, transform = None, *args, **kwargs):
        self.transform = lambda x: x
        if transform is not None:
            self.transform = transform
        self.kwargs = kwargs
        self.args = args
        self.levy_stable = LevyStable()
        self._data = None
        if dataset == 'skewed_levy':
            self.state_space = 'continuous'
        elif dataset == 'sas':
            self.state_space = 'continuous'
        elif dataset == 'gmm_2':
            self.state_space = 'continuous'
        elif dataset == 'gmm_grid':
            self.state_space = 'continuous'
        elif dataset == 'swiss_roll':
            self.state_space = 'continuous'
        elif dataset == 'sas_grid':
            self.state_space = 'continuous'
        elif dataset == 'saws_1d':
            self.state_space = 'discrete'
        elif dataset == 'uniform_saws_1d':
            self.state_space = 'discrete'
        else:
            raise Exception('Unknown distribution to sample from. \
            Available distributions: {}'.format(list(self.available_distributions.keys())))

        self.generator = self.available_distributions[dataset]
    
    
    def setTransform(self, transform):
        self.transform = transform
    
    # replaces missing elements of args and kwargs by those of self.args and self.kwargs
    # replaces elements of kwargs totally if non void
    def setParams(self, *args, **kwargs):
        if args == () and kwargs == {}:
            raise Exception('Given void parameters')
        
        self.args = tuple(map(lambda x, y: y if y is not None else x, self.args, args))
        self.kwargs.update(kwargs)

    def getSignature(self):
        return signature(self.generator)
    
    def getName(self):
        return 
    
    # generate by replacing self.args by args if it is not ()
    # and replace potentially missing kwargs by thos provided.
    def generate(self, *args, **kwargs):
        tmp_kwargs = self.kwargs | kwargs        
        if args == () and kwargs == {} and self.kwargs == {}:
            raise Exception('No parameters for data generation')
        if args == ():
            self._data = self.transform(self.generator(*self.args, **tmp_kwargs))
            return self._data
        self._data = self.transform(self.generator(*args, **tmp_kwargs))
        return self._data
    
    def refresh_data(self):
        self._data = self.transform(self.generator(*self.args, **self.kwargs))

    def __len__(self):
        return self._data.shape[0]
    
    def __getitem__(self, idx):
        return self._data[idx]




