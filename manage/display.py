import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import torch



def _img_to_plt_img(tensor):
    return torch.stack([tensor[i]for i in range(tensor.shape[0])], dim = -1)

def _get_image_from(sample,
                    black_and_white = False):
    img = sample
    img = _img_to_plt_img(img)
    # potentially repeat last dimension for signle channel data to be black and white
    if black_and_white and img.shape[-1] == 1:
        img = img.repeat(1, 1, 3)
    return img


def get_image(sample,
              black_and_white=False, # in the case of single channel data
              title=None):
    img = _get_image_from(sample, 
                        black_and_white=black_and_white)
    fig = plt.figure()
    plt.imshow(img, animated = False)
    if title is not None:
        plt.title(title)
    return fig


def get_plot(generated_data,
            original_data = None,
            figsize = (5, 5),
            **plot_kwargs):
    gen_data = generated_data
    gen_data = gen_data.squeeze(1) # remove channel
    fig = plt.figure(figsize=figsize)
    if original_data is not None:
        tmp_original_data = original_data.squeeze(1) # remove channel since simple 2d data
        _plot_data(tmp_original_data, 
                   label='Original data', 
                   **plot_kwargs)
    _plot_data(gen_data, 
               label='Generated data', 
               **plot_kwargs)
    plt.legend()
    return fig

def _get_scatter_marker_specific_kwargs(marker):
    if marker == '.':
        return {'marker': marker, 'lw': 0, 's': 1}
    return {'marker': marker}


def _plot_data(data, 
               ax = None,
               **plot_kwargs):
    assert data.shape[1] == 2, 'only supports plotting 2d data'
    canvas = plt if ax is None else ax 
    fig = canvas.scatter(data[:, 0], data[:, 1], **plot_kwargs) # _get_scatter_marker_specific_kwargs(marker)
    return fig

def get_animation(method,
                generated_data_history,
                original_data = None,
                is_image = False,
                xlim = (-1.5, 1.5),
                ylim = (-1.5, 1.5),
                title = None,
                figsize = (5, 5),
                **plot_kwargs):
    
    assert method is not None, 'Must give method object to determine the time spacing'
    assert not (is_image and (original_data is not None)), 'Cannot plot original data for image data'
    
    
    print('Generating animation, with {} frames'.format(generated_data_history.shape[0]))
    
    # determine the timesteps we are working with. Add the last value T to the array
    # must reverse the list to have it in increasing order
    # len(self.history) == T+1 (goes from x_T to ... x_0)
    timesteps = np.array([x for x in method.get_timesteps(generated_data_history.shape[0]-1)])
    T = timesteps[-1]
    # timesteps = timesteps[::-1].copy()
    timesteps = torch.tensor(timesteps)

    num_frames = 60*3

    if original_data is not None:
        original_data = original_data.squeeze(1)
    
    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and axes once.
    #if title is not None:
    #    plt.title(title)
    if is_image:
        image_shape = _img_to_plt_img(generated_data_history[0]).shape
        im = plt.imshow(np.random.random(image_shape), interpolation='none')
    else:
        scatter = ax.scatter([], [], **plot_kwargs) #, **_get_scatter_marker_specific_kwargs(marker))
        scatter_orig = ax.scatter([], [], **plot_kwargs, color='orange') #, **_get_scatter_marker_specific_kwargs(marker))

    def init_frame_2d():
        #ax.clear()  # Clear the current axes.
        ax.set_xlim(xlim)  # Set the limit for x-axis.
        ax.set_ylim(ylim)  # Set the limit for y-axis.
        ax.set_title(title)
        scatter.set_offsets(np.empty((0, 2)))  # Properly shaped empty array
        scatter_orig.set_offsets(np.empty((0, 2)))  # Properly shaped empty array
        return scatter, scatter_orig, 
    
    def init_frame_image():
        im.set_data(np.random.random(image_shape))
        return im, 

    def get_interpolation_values(i):
        t = T * (i / (num_frames - 1))
        k = torch.searchsorted(timesteps, t) - 1
        if k < 0:
            k = 0
        if k >= len(timesteps)-1:
            k = len(timesteps) - 2
        l = (t - timesteps[k]) / (timesteps[k+1] - timesteps[k])
        return k, l
        Xk1 = self.history[k+1].cpu().squeeze(1)[:limit_nb_datapoints]
        Xk = self.history[k].cpu().squeeze(1)[:limit_nb_datapoints]
        Xvis = Xk1 * l + Xk* (1 - l)
        return Xvis
    
    def draw_frame_2d(i):
        #ax.clear()
        k, l = get_interpolation_values(i)
        Xk1 = generated_data_history[k+1].cpu().squeeze(1)
        Xk = generated_data_history[k].cpu().squeeze(1)
        Xvis = Xk1 * l + Xk* (1 - l)
        scatter.set_offsets(Xvis)
        if original_data is not None:
            scatter_orig.set_offsets(original_data)
            return scatter, scatter_orig, 
        return scatter, 

    def draw_frame_image(i):
        k, l = get_interpolation_values(i)
        Xk1 = generated_data_history[k+1][0].cpu() # just take first element of the batch. batch size should always be one anyway
        Xk = generated_data_history[k][0].cpu() # just take first element of the batch. batch size should always be one anyway
        Xvis = Xk1 * l + Xk* (1 - l)
        img = _get_image_from([Xvis], black_and_white=True)
        im.set_data(img)
        return im,

    # 3000 ms per loop
    if is_image:
        anim = animation.FuncAnimation(fig, draw_frame_image, frames=num_frames, interval= 3000 / num_frames, blit=True, init_func=init_frame_image)
    else:
        anim = animation.FuncAnimation(fig, draw_frame_2d, frames=num_frames, interval= 3000 / num_frames, blit=True, init_func=init_frame_2d)
    return anim






