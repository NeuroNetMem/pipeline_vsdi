"""
Module for Nonlinear Dimensionality Reduction.

This module provides functionalities for training, analyzing and performing various operations with Convolutional Autoencoders (Conv_VAE). 
The module is specifically designed for the purpose of training a convolutional autoencoder to compress image data into a low-dimensional latent 
representation and evaluate the quality of these representations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from random import shuffle


class Conv_AE(nn.Module):
    """
    Convolutional Autoencoder (Conv_AE) Class.

    This class implements a convolutional autoencoder using PyTorch's nn.Module. 
    It includes an encoder and a decoder. The encoder compresses the input data 
    into a latent space, and the decoder reconstructs the original data from 
    the latent representation. The model is trained to minimize the difference 
    between the input and the output of the autoencoder.
    """

    def __init__(self, latent_dim, h_pixels, w_pixels):
        """
        Initializes the Conv_AE class.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent space.
        """
        super(Conv_AE, self).__init__()
        self.latent_dim = latent_dim
        self.h_pixels = h_pixels
        self.w_pixels = w_pixels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Linear layer for mean and variance
        self.fc = nn.Linear(64*self.h_pixels*self.w_pixels, self.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64 * self.h_pixels * self.w_pixels),
            nn.ReLU(),
            nn.Unflatten(1, (64, self.w_pixels, self.h_pixels)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Passes the input through the encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            The input to the autoencoder.

        Returns
        -------
        reconstructed : torch.Tensor
            The reconstructed input produced by the decoder.

        h : torch.Tensor
            The latent representation of the input.
        """
        # Encoder
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)

        # Reparameterization
        h = self.fc(encoded)

        # Decoder
        reconstructed = self.decoder(h)

        return reconstructed


def train_autoencoder(model, trainloader, validloader=None,
                      epochs=5, criterion=nn.MSELoss(),
                      optimizer=None, lr=pow(10, -4),
                      weight_decay=0,
                      early_stopping=False):

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)

    train_history = []
    valid_history = []

    for e in range(epochs):
        train_loss = 0.0
        model.train()  # set model in train mode
        with tqdm(total=len(trainloader)) as pbar:
            for x in trainloader:
                if torch.cuda.is_available():
                    x = x.cuda()

                optimizer.zero_grad()
                target = model(x)
                loss = criterion(target, x)
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                train_loss += batch_loss
                train_history.append(batch_loss)

                pbar.update(1)
                pbar.set_description(
                    f"Epoch {e+1}/{epochs}, batch loss: {batch_loss:.4f}")

        # if validation loader is provided, use it at end of epoch
        if validloader is not None:
            valid_loss = 0.0
            model.eval()     # Optional when not using Model Specific layer
            for x in validloader:
                if torch.cuda.is_available():
                    x = x.cuda()

                target = model(x)
                loss = criterion(target, x)
                valid_loss = loss.item() * x.size(0)

            valid_history.append(valid_loss / len(validloader))

            print(
                f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')

        if e > 1 and early_stopping:
            if valid_history[-2] < valid_history[-1]:
                print(
                    f'Validation increased({valid_history[-2]:.6f}--->{valid_history[-2]:.6f}) \t stopping training')
                break

    return {'train_history': np.asarray(train_history), 'validation_history': np.asarray(valid_history)}


def get_clamped_modes(model):

    input = torch.eye(model.latent_dim)
    imgs = model.decoder(input)
    imgs = imgs.detach().numpy()
    imgs = imgs[:, 0, :, :]

    return imgs


def plot_latent_modes(imgs, mask):

    fig, ax = plt.subplots(1, len(imgs), figsize=(5*len(imgs), 5))
    for i, img in enumerate(imgs):
        plot_img = img.copy()
        plot_img[mask[:, :-1] == 0] = np.nan
        ax[i].imshow(plot_img, cmap=plt.cm.gnuplot2)
        ax[i].axis('off')

    plt.tight_layout()
