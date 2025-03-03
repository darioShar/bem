import torch



# Quantiles
# msle_quantiles =  torch.concat((
#     torch.linspace(0.0001, 0.01, 50),
#     torch.linspace(0.01, 0.99, 99),
#     torch.linspace(0.99, 0.9999, 50),
# ))

msle_quantiles =  torch.tensor([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999])#torch.linspace(0.001, 0.999, 999)


def get_msle_empirical(
                original_data, 
                generated_data, 
                xi_msle = 0.975,
                agg = '1d_projection'):
    assert agg in ['mean', 'sum', '1d_projection'], 'agg must be either mean, sum or 1d_projection'
    
    def compute_msle_from_empirical_quantiles( 
                                data_1,
                                data_2,
                                xi_msle = 0.95):
        # Compute the quantiles
        generated_quantile_values = torch.quantile(data_1, msle_quantiles)
        original_quantile_values = torch.quantile(data_2, msle_quantiles)
        value_msle = msle_quantiles>xi_msle #int(xi_msle*original_quantile_values.numel())
        y_msle = torch.abs(torch.log(generated_quantile_values[value_msle]) - torch.log(original_quantile_values[value_msle]))
        y_msle = (y_msle**2).mean()
        return y_msle

    if '1d_projection' == agg:
        # original_data = original_data.flatten()
        # generated_data = generated_data.flatten()
        return compute_msle_from_empirical_quantiles(
            original_data[:, 0, 0], 
            generated_data[:, 0, 0], 
            xi_msle)
    else:
        msles = torch.stack( [
            compute_msle_from_empirical_quantiles(
                original_data[:, 0, i],
                generated_data[:, 0, i],
                xi_msle,
            )
            for i in range(original_data.shape[-1])]
        )
        return torch.mean(msles) if 'mean' == agg else torch.sum(msles)
        


def get_msle_truth_vs_empirical(
                true_ppf,
                generated_data, 
                xi_msle = 0.95,
                agg = '1d_projection'):
    assert agg in ['mean', 'sum', '1d_projection'], 'agg must be either mean, sum or 1d_projection'

    real_quantiles = true_ppf(msle_quantiles)
    
    def compute_msle_real_and_generated_quantiles( 
                                data,
                                xi_msle = 0.95):
        # Compute the quantiles
        generated_quantile_values = torch.quantile(data, msle_quantiles)
        value_msle = msle_quantiles>xi_msle #int(xi_msle*original_quantile_values.numel())
        y_msle = torch.abs(torch.log(generated_quantile_values[value_msle]) - torch.log(real_quantiles[value_msle]))
        y_msle = (y_msle**2).mean()
        return y_msle

    if agg == '1d_projection':
        return compute_msle_real_and_generated_quantiles(
            generated_data[:, 0, 0], 
            xi_msle)
    else:
        msles = torch.stack( [
            compute_msle_real_and_generated_quantiles(
                generated_data[:, 0, i],
                xi_msle,
            )
            for i in range(generated_data.shape[-1])]
        )
        return torch.mean(msles) if agg == 'mean' else torch.sum(msles)


