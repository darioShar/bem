import numpy as np
import os
from PIL import Image
import torch




def inf_train_gen(img_name, data_size):
    def gen_data_from_img(image_mask, train_data_size):
        def sample_data(train_data_size):
            inds = np.random.choice(
                int(probs.shape[0]), int(train_data_size), p=probs)
            m = means[inds] 
            samples = np.random.randn(*m.shape) * std + m 
            return samples
        img = image_mask
        h, w = img.shape
        xx = np.linspace(-4, 4, w)
        yy = np.linspace(-4, 4, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        means = np.concatenate([xx, yy], 1) # (h*w, 2)
        img = img.max() - img
        probs = img.reshape(-1) / img.sum() 
        std = np.array([8 / w / 2, 8 / h / 2])
        full_data = sample_data(train_data_size)
        return full_data
    image_mask = np.array(Image.open(f'{img_name}.png').rotate(
        180).transpose(0).convert('L'))
    dataset = gen_data_from_img(image_mask, data_size)
    return dataset / 4


def check_png_dataset_exists(data_path, img_name):
    img_path = os.path.join(data_path, "{}".format(img_name))
    return os.path.exists(img_path)

def get_dataset_from_png(data_path, img_name, nsamples):
    img_path = os.path.join(data_path, "{}".format(img_name))
    
    xraw = inf_train_gen(
        img_path,
        nsamples
    )
    xte = inf_train_gen(
        img_path,
        nsamples
    )
    # add channel dimension
    xraw = torch.from_numpy(xraw).float().unsqueeze(1)
    xte = torch.from_numpy(xte).float().unsqueeze(1)
    
    # create datasets
    dataset = torch.utils.data.TensorDataset(xraw, torch.tensor([0.]).repeat(xraw.shape[0]))
    test_dataset = torch.utils.data.TensorDataset(xte, torch.tensor([0.]).repeat(xte.shape[0]))
    
    return dataset, test_dataset