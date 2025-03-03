import numpy as np
import torch
import pyemd

# to compute W_p loss
def compute_mean_lploss(tens, lploss = 2.):
            # simply flatten the tensor
            return torch.pow(torch.linalg.norm(tens.flatten(), 
                                               ord = lploss,), 
                                               #dim = list(range(0, len(tens.shape)))), 
                                                lploss) \
                / torch.prod(torch.tensor(tens.shape), dim = 0)


def compute_sliced_wasserstein(
    data: torch.Tensor,
    model_data: torch.Tensor,
    n_projections: int = 500
) -> float:
    # Ensure the data is on CPU for numpy-based Wasserstein, if your function relies on numpy
    data = data.float().cpu()
    model_data = model_data.float().cpu()

    N, d = data.shape
    
    # Container for distances across projections
    sw_values = []

    for _ in range(n_projections):
        # Sample a random direction in the d-simplex:
        # we draw d independent exponentials (or uniform(0,1) > 0) and normalize
        direction = torch.rand(d - 1)
        # add 0 and 1 to direction
        direction = torch.cat([torch.tensor([0.0]), direction, torch.tensor([1.0])])
        direction = torch.sort(direction).values
        direction = direction[1:] - direction[:-1]
        
        # Project data onto this direction
        proj_data = (data * direction).sum(dim=1).numpy()
        proj_model = (model_data * direction).sum(dim=1).numpy()
        
        # proj_data and proj_model are in [0, 1]
        
        # Compute the 1D Wasserstein distance of the projections
        w = compute_wasserstein_distance(proj_data, proj_model, bins='auto')
                                            # bins = 250 if N >= 512 else 'auto')
        sw_values.append(w)

    # Average the 1D Wasserstein distances
    return float(sum(sw_values) / len(sw_values))



# updated to manually compute bins if requested
def compute_wasserstein_distance(data, 
                                 gen_samples,
                                 manual_compute = False,
                                 num_samples = -1, 
                                 distance='euclidean',
                                 normalized=True,
                                 bins='auto',
                                 _range=None):
    if manual_compute:
        # each vector is a histogram fof bins
        # that is the density of the corresponding data in each bin.
        # pairwise distance is computed between each bin
        # thus if each data point has its own bin, we need 2*N bins
        # and a 4*N*N matrix.
        # use L1 loss
        lploss = 1.
        N = np.min((data.shape[0], gen_samples.shape[0])) if num_samples == -1 else num_samples
        # equal histograms, first bins for data, second one for gen_samples
        data_1_array = np.concatenate((np.ones(N) / N, np.zeros(N)),dtype = np.float64)
        data_2_array = np.concatenate((np.zeros(N), np.ones(N) / N),dtype = np.float64)
        distance_data = np.zeros((2*N, 2*N), dtype = np.float64)
        for i in range(2*N):
            for j in range(2*N):
                #distance_data = np.array([[compute_loss_all(data1[i] - data2[j]) for j in range(data2.shape[0])] for i in range(data1.shape[0])])
                data_i = data[i] if i < N else gen_samples[i % N]
                data_j = data[j] if j < N else gen_samples[j % N]
                distance_data[i, j] = compute_mean_lploss(data_i - data_j, lploss = lploss)
        res = pyemd.emd(data_1_array,
                        data_2_array,
                        np.float64(distance_data))
        res = res**(lploss)
        return res
    return pyemd.emd_samples(gen_samples[:num_samples], 
                             data[:num_samples],
                            distance = distance, 
                            normalized=normalized, 
                            bins=bins, 
                            range=_range)