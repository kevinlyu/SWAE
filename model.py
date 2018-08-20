import torch.nn as nn

import torch.nn.functional as F


class Autoencoder(nn.Module):

    def __init__(self, in_channels=16, lrelu_slope=0.2, fc_dim=128, encoded_dim=2):
        super(Autoencoder, self).__init__()

        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.fc_dim = fc_dim
        self.encoded_dim = encoded_dim

        # encoder part
        self.encoder = nn.Sequential(
            nn. Conv2d(1, self.in_channels*1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*1, self.in_channels *
                      1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.in_channels*1, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*2, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.in_channels*2, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.AvgPool2d(kernel_size=2, padding=1)
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.in_channels*4*4*4, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.encoded_dim)
        )

        # decoder part
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.encoded_dim, self.fc_dim),
            nn.Linear(self.fc_dim, self.in_channels*4*4*4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=0),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*4, self.in_channels *
                      4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels*4, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*2, self.in_channels *
                      2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope),

            nn.Conv2d(self.in_channels*2, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(-1, self.in_channels*4*4*4)
        z = self.encoder_fc(z)

        x = self.decoder_fc(z)
        x = x.view(-1, 4*self.in_channels, 4, 4)
        x = self.decoder(x)
        x = F.sigmoid(x)

        return x, z

        