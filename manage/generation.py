import torch
import copy
#from IPython.display import HTML


class GenerationManager:
    
    # same device as pdmp
    def __init__(self, 
                 method,
                 dataloader,
                 is_image,
                 **kwargs
                 ):
        self.method = method
        self.original_data = dataloader
        self.is_image = is_image
        self.kwargs = kwargs
        self.samples = []
        self.history = []

    def generate(self,
                 models,
                 nsamples,
                 get_sample_history = False, # if method supports it, get the history of the samples
                 print_progression=False,
                 **kwargs
                 ):
        assert nsamples > 0, 'nsamples must be greater than 0, got {}'.format(nsamples)
        tmp_kwargs = copy.deepcopy(self.kwargs)
        tmp_kwargs.update(kwargs)

        _, (data) = next(enumerate(self.original_data))
        if self.is_image:
            data, y = data
        size = list(data.size())
        size[0] = nsamples
        x = self.method.sample(shape=size,
                            models = models,
                            print_progression = print_progression,
                            get_sample_history = get_sample_history,
                            is_image = self.is_image,
                            **tmp_kwargs)
        
        if get_sample_history:
            self.history = x.cpu()
            self.samples = self.history[-1, ...]
        else:
            self.samples = x.cpu()


    def load_original_data(self, nsamples):
        data_size = 0
        total_data = torch.tensor([])
        while data_size < nsamples:
            _, (data) = next(enumerate(self.original_data))
            if self.is_image:
                data, y = data
            total_data = torch.concat([total_data, data])
            data_size += data.size()[0]
        return total_data[:nsamples]
    
