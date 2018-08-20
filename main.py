from model import Autoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from distributions import *
import torch.nn.functional as F

'''
Implementation of Sliced Wasserstein Autoencoder (SWAE)
Reference: https://github.com/eifuentes/swae-pytorch
'''
def get_theta(embedding_dim, num_samples=50):
    theta = [w/np.sqrt((w**2).sum())
             for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor)


def sliced_wasserstein_distance(encoded_samples, distribution_fn=random_circle, num_projections=50, p=2):

    batch_size = encoded_samples.size(0)
    z_samples = distribution_fn(batch_size)
    embedding_dim = z_samples.size(1)

    theta = get_theta(embedding_dim, num_projections)
    proj_ae = encoded_samples.matmul(theta.transpose(0, 1))
    proj_z = z_samples.matmul(theta.transpose(0, 1))
    w_distance = torch.sort(proj_ae.transpose(0, 1), dim=1)[
        0]-torch.sort(proj_z.transpose(0, 1), dim=1)[0]

    w_distance_p = torch.pow(w_distance, p)

    return w_distance_p.mean()


class SAE:

    def __init__(self, autoencoder, optimizer, distribution_fn, num_projections=50, p=2, weight_swd=10):
        self.model = autoencoder
        self.optimizer = optimizer
        self.distribution_fn = distribution_fn
        self.embedding_dim = self.model.encoded_dim
        self.num_projections = num_projections
        self.p = p
        self.weight_swd = weight_swd

    def train(self, x):
        self.optimizer.zero_grad()
        x.cuda()
        recon_x, z = self.model(x)
        
        l1 = F.l1_loss(recon_x, x)
        bce = F.binary_cross_entropy(recon_x, x)

        recon_x = recon_x.cpu()
        z = z.cpu()

        w2 = float(self.weight_swd)*sliced_wasserstein_distance(z,
                                                              self.distribution_fn, self.num_projections, self.p)
        w2=w2.cuda()
        loss = l1+bce+w2

        loss.backward()
        self.optimizer.step()

        return {'loss': loss, 'bce': bce, 'l1': l1, 'w2': w2, 'encode': z, 'decode': recon_x}



mnist = torch.utils.data.DataLoader(datasets.MNIST("./mnist/", train=True, download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor()
                                                   ])), batch_size=128, shuffle=True)
cudnn.benchmark = True
ae = Autoencoder().cuda()
print(ae)
critetion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters())

total_epoch = 20


trainer = SAE(ae, optimizer, random_circle)
ae.train()


for epoch in range(total_epoch):

    for index, (img, label) in enumerate(mnist):
        img = img.cuda()
        batch_result = trainer.train(img)
        if (index+1) % 10 == 0:
            print("{:.4f}".format(batch_result["loss"]))
