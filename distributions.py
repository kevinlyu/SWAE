import numpy as np
import torch
from sklearn.datasets import make_circles

# distribution to sample from


def random_circle(batch_size):
    r = np.random.uniform(size=batch_size)
    theta = 2*np.pi*np.random.uniform(size=batch_size)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = np.array([x, y]).T
    return torch.from_numpy(z).type(torch.FloatTensor)


def random_ring(batch_size):
    t = make_circles(2*batch_size, noise=0.01)
    z = np.squeeze(t[0][np.argwhere(t[1] == 0), :])
    return torch.from_numpy(z).type(torch.FloatTensor)


def random_uniform(batch_size):
    z = 2*(np.random.uniform(size=(batch_size, 10))-0.5)
    return torch.from_numpy(z).type(torch.FloatTensor)
