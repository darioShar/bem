import torch

def compute_kl_div_and_hellinger(data, gen_samples):
    # compute kl divergence and hellinger distance
    # generate all unique bits values
    d = data.shape[1]
    bit_values = [bin(i)[2:].zfill(d) for i in range(2**d)]
    # convert value to tensor
    tensor_bit_values = torch.stack([torch.tensor([int(v) for v in value]) for value in bit_values])

    data_values, data_counts = torch.unique(data, dim=0, return_counts=True)
    model_values, model_counts = torch.unique(gen_samples, dim=0, return_counts=True)

    p_data = torch.zeros(len(tensor_bit_values))
    p_model = torch.zeros(len(tensor_bit_values))
    for i, value in enumerate(tensor_bit_values):
        data_count = 0
        model_count = 0
        val_mask_data = (data_values == value).all(dim=1)
        if val_mask_data.any():
            data_count = data_counts[val_mask_data].item()

        val_mask_model = (model_values == value).all(dim=1)
        if val_mask_model.any():
            model_count = model_counts[val_mask_model].item()
        p_data[i] = data_count
        p_model[i] = model_count

    p_data = p_data / p_data.sum()
    p_model = p_model / p_model.sum()
    
    # kl divergence
    kl_div = torch.nn.functional.kl_div(torch.log(p_model), p_data, reduction='sum')
    # hellinger distance
    hellinger = torch.sqrt(0.5 * ((torch.sqrt(p_model) - torch.sqrt(p_data))**2).sum())
    
    return kl_div.item(), hellinger.item()