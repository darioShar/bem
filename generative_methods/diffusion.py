import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms


#############################################
# Helper function 
#############################################

def match_last_dims(data, size):
    """
    Repeat a 1D tensor so that its last dimensions [1:] match `size[1:]`.
    Useful for working with batched data.
    """
    assert len(data.size()) == 1, "Data must be 1-dimensional (one value per batch)"
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))

class DiffusionProcess:
    def __init__(self, 
                 device=None, 
                 T=1.0, 
                 process_type='VP', 
                 schedule='cosine',
                 rescale_timesteps=True,
                 learn_variance=False,
                 **kwargs):
        """
        process_type: 'VP' (variance preserving) or 'VE' (variance exploding)
        schedule: for VP, choose 'linear' or 'cosine'
        T: time horizon (used to sample t ~ Uniform(0,T)); the neural net always receives normalized time in [0,1]
        """
        self.device = 'cpu' if device is None else device
        self.T = T
        self.process_type = process_type
        self.schedule = schedule
        self.learn_variance = learn_variance
        self.rescale_timesteps = rescale_timesteps
        
        assert self.rescale_timesteps == True, "rescale_timesteps must be set to True"
        assert self.learn_variance == False, "learn_variance is not implemented yet"
        
        if process_type == 'VP':
            if schedule == 'linear':
                self.beta_min = kwargs.get('beta_min', 0.1)
                self.beta_max = kwargs.get('beta_max', 20.0)
            elif schedule == 'cosine':
                # s is a small offset (default 0.008) as in Nichol & Dhariwal’s cosine schedule.
                self.s = kwargs.get('s', 0.008)
            else:
                raise ValueError("Unknown VP schedule type")
        elif process_type == 'VE':
            self.sigma_min = kwargs.get('sigma_min', 0.01)
            self.sigma_max = kwargs.get('sigma_max', 50.0)
        else:
            raise ValueError("Unknown process type: {}".format(process_type))
    
    def alpha_bar(self, t_norm):
        """
        For VP processes, returns the cumulative product (or survival probability) at normalized time t_norm.
        For linear: ᾱ(t) = exp( - [β_min t + 0.5 (β_max - β_min)t^2] )
        For cosine: ᾱ(t) = cos( ((t + s)/(1+s))*(π/2) )^2
        """
        if self.process_type == 'VP':
            if self.schedule == 'linear':
                integrated_beta = self.beta_min * t_norm + 0.5 * (self.beta_max - self.beta_min) * t_norm**2
                return torch.exp(-integrated_beta)
            elif self.schedule == 'cosine':
                return torch.cos((t_norm + self.s) / (1 + self.s) * (torch.pi / 2))**2
        else:
            return None

    def sigma_fn(self, t_norm):
        """
        For VE processes, returns the noise scale at normalized time t_norm.
        Using an exponential schedule: σ(t) = σ_min * (σ_max/σ_min)^t
        """
        if self.process_type == 'VE':
            return self.sigma_min * (self.sigma_max / self.sigma_min)**(t_norm)
        else:
            return None
        
    def get_timesteps(self, N):
        return torch.linspace(self.T, 1e-6, N + 1)
    
    def get_score_from_eps(self, eps, t):
        """
        Given the predicted noise eps, returns the score (i.e. ∇_x log p_t(x)).
        For VP: score = - (predicted noise) / sqrt(1 - ᾱ(t))
        For VE: score = - (predicted noise) / σ(t)
        """
        t_norm = t / self.T
        if self.process_type == 'VP':
            alpha_bar = self.alpha_bar(t_norm).view(-1, *([1] * (eps.dim() - 1)))
            score = -eps / torch.sqrt(1 - alpha_bar)
            return score
        elif self.process_type == 'VE':
            sigma_t = self.sigma_fn(t_norm).view(-1, *([1] * (eps.dim() - 1)))
            score = -eps / sigma_t
            return score
    
    def score_fn(self, model, x, t, **model_kwargs):
        """
        Given the noise-predicting model, returns the score (i.e. ∇_x log p_t(x))
        at actual time t. Note that the model expects a normalized time (t/T).
        For VP: score = - (predicted noise) / sqrt(1 - ᾱ(t))
        For VE: score = - (predicted noise) / σ(t)
        """
        t_norm = t / self.T  # normalize to [0,1]
        if self.process_type == 'VP':
            alpha_bar = self.alpha_bar(t_norm).view(-1, *([1] * (x.dim() - 1)))
            epsilon = model(x, t_norm, **model_kwargs) #.view(-1, 1))
            score = -epsilon / torch.sqrt(1 - alpha_bar)
            return score
        elif self.process_type == 'VE':
            sigma_t = self.sigma_fn(t_norm).view(-1, *([1] * (x.dim() - 1)))
            epsilon = model(x, t_norm, **model_kwargs)#.view(-1, 1))
            score = -epsilon / sigma_t
            return score

    def forward(self, x_start, t_norm):
        """
        Forward (diffusion) process: given a clean sample x_start and time t (in [0,T]),
        returns the noised version x_t.
        For VP: x_t = sqrt(ᾱ(t)) x_start + sqrt(1-ᾱ(t)) noise
        For VE: x_t = x_start + σ(t)*noise
        """
        noise = torch.randn_like(x_start)
        if self.process_type == 'VP':
            alpha_bar = self.alpha_bar(t_norm).view(-1, *([1] * (x_start.dim() - 1)))
            x_t = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        elif self.process_type == 'VE':
            sigma_t = self.sigma_fn(t_norm).view(-1, *([1] * (x_start.dim() - 1)))
            x_t = x_start + sigma_t * noise
        return x_t, noise

    def training_losses(self, models, x_start, model_kwargs=None, **kwargs):
        """
        Training loss for the diffusion process.
        Samples t ~ Uniform(0, T), applies the forward process, and then
        computes the MSE loss between the network’s predicted noise and the true noise.
        """
        model = models['default']
        x_start = x_start.to(self.device)
        batch_size = x_start.size(0)
        # Sample t uniformly from [0, T]
        t = torch.rand(batch_size, device=self.device) * self.T
        t_norm = t / self.T
        x_t, noise = self.forward(x_start, t_norm)
        
        if model_kwargs is None:
            model_kwargs = {}
        # The model takes x_t and normalized time t_norm
        predicted_noise = model(x_t, t_norm, **model_kwargs)
        loss = F.mse_loss(predicted_noise, noise)
        
        score_loss = F.mse_loss(
            self.get_score_from_eps(predicted_noise, t), 
            self.get_score_from_eps(noise, t)
            )
                
        return {'loss': loss, 'score_loss': score_loss}

    def sample(self, 
               models, 
               shape, 
               reverse_steps=1000, 
               epsilon = 1e-4,
               guidance_scale = 1.0,
               get_sample_history=False, 
               progress=True, 
               deterministic = False,
               model_kwargs = None,
               **kwargs):
        """
        SDE sampling using the Euler–Maruyama method to solve the reverse-time SDE:
          dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dẆ
        For VP:
          f(x,t) = -0.5 β(t)x  and  g(t) = sqrt(β(t))
          where β(t) is given either by a linear or cosine schedule.
        For VE:
          f(x,t) = 0  and  g(t) = σ(t)
        """
        model = models['default']
        if model_kwargs is None:
            model_kwargs = {}
        # Initialize x_T (the prior sample)
        if self.process_type == 'VP':
            xt = torch.randn(shape, device=self.device)
        elif self.process_type == 'VE':
            t_norm = torch.tensor(1.0, device=self.device)
            sigma_T = self.sigma_fn(t_norm).view(*([1] * len(shape)))
            xt = sigma_T * torch.randn(shape, device=self.device)
        samples = []
        model.eval()
        with torch.inference_mode():
            # Create a time discretization from T to 0
            t_seq = torch.linspace(self.T, epsilon, reverse_steps + 1, device=self.device)
            if progress:
                progress_bar = tqdm(range(reverse_steps))
            else:
                progress_bar = range(reverse_steps)
            for i in progress_bar:
                t_current = t_seq[i]
                t_next = t_seq[i + 1]
                dt = t_next - t_current  # dt is negative (reverse time)
                # Create a batch of current time values for the update.
                t_batch = torch.full((shape[0],), t_current, device=self.device)
                t_norm_batch = t_batch / self.T

                if self.process_type == 'VP':
                    # Compute β(t) depending on schedule.
                    if self.schedule == 'linear':
                        beta_t = self.beta_min + t_norm_batch * (self.beta_max - self.beta_min)
                    elif self.schedule == 'cosine':
                        val = ((t_norm_batch + self.s) / (1 + self.s)) * (torch.pi / 2)
                        beta_t = (torch.pi / ((1 + self.s))) * torch.sin(val)
                    beta_t = beta_t.view(-1, *([1] * (xt.dim() - 1)))
                    f = -0.5 * beta_t * xt
                    g = torch.sqrt(beta_t)
                elif self.process_type == 'VE':
                    sigma_t = self.sigma_fn(t_norm_batch).view(-1, *([1] * (xt.dim() - 1)))
                    f = 0.0
                    g = sigma_t
                
                
                # Get the score (using the noise-predicting network)
                score = self.score_fn(model, xt, t_batch, **model_kwargs)
                # Euler–Maruyama update:
                #   x = x + [f - g^2 * score] dt + g * sqrt(-dt) * z,   where z ~ N(0, I)
                if not deterministic:
                    z = torch.randn_like(xt)
                    xt = xt + (f - (g**2) * score) * dt + g * torch.sqrt(-dt) * z
                else:
                    xt = xt + (f - (g**2) * score / 2) * dt
                if get_sample_history:
                    samples.append(xt.clone())
        return xt if not get_sample_history else torch.stack(samples)

