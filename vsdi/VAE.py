"""
Module for Nonlinear Dimensionality Reduction.

This module provides functionalities for training, analyzing and performing various operations with Convolutional Autoencoders (Conv_AE). The module is specifically designed for the purpose of training a convolutional autoencoder to compress image data into a low-dimensional latent representation and evaluate the quality of these representations.
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
    def __init__(self, latent_dim):
        """
        Initializes the Conv_AE class.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent space.
        """
        super(Conv_AE, self).__init__()
        self.latent_dim = latent_dim

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
        self.fc = nn.Linear(64 * 84 * 48, self.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64 * 84 * 48),
            nn.ReLU(),
            nn.Unflatten(1, (64, 84, 48)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()
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

        return reconstructed, h

    def backward(self, optimizer, criterion, x, y_true):
        """
        Computes the loss, performs backpropagation, and updates the model parameters.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer used to update the model parameters.
            
        criterion : torch.nn.modules.loss._Loss
            The loss function used to measure the difference between the 
            reconstructed input and the original input.
        
        x : torch.Tensor
            The input to the autoencoder.
            
        y_true : torch.Tensor
            The true values (same as the input for an autoencoder).
            
        Returns
        -------
        loss.item() : float
            The computed loss value.
        """
        optimizer.zero_grad()
        y_pred, _ = self.forward(x)
        mse = criterion(y_pred, y_true)
        loss = mse
        loss.backward()
        optimizer.step()
        return loss.item()


def create_dataloader(dataset, batch_size=128, reshuffle_after_epoch=True):
    '''
    Creates a DataLoader for Pytorch to train the autoencoder with the image data converted to a tensor.

    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width).
        batch_size (int; default=32): the size of the batch updates for the autoencoder training.

    Returns:
        DataLoader (Pytorch DataLoader): dataloader that is ready to be used for training an autoencoder.
    '''
    if dataset.shape[-1] == 3:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=reshuffle_after_epoch)


def train_autoencoder(model, train_loader, dataset=[], num_epochs=1000, learning_rate=1e-3, L2_weight_decay=0):
    """
    Trains the autoencoder model.

    This function trains the autoencoder model using the Adam optimizer 
    and the Mean Squared Error loss function. It also tracks the history 
    of loss values for each epoch, and optionally, the evolution of the 
    latent vectors.

    Parameters
    ----------
    model : Conv_AE
        The autoencoder model to be trained.

    train_loader : torch.utils.data.DataLoader
        The data loader that provides batches of training data.

    dataset : list, optional
        If provided, this function will also track the evolution of the 
        latent vectors for each data point in this dataset.

    num_epochs : int, optional
        The number of epochs for training the model. Default is 1000.

    learning_rate : float, optional
        The learning rate for the Adam optimizer. Default is 1e-3.

    L2_weight_decay : float, optional
        The weight decay (L2 penalty) for the Adam optimizer. Default is 0.

    Returns
    -------
    history : list
        A list containing the history of loss values for each epoch.

    embeddings : numpy.ndarray
        A numpy array containing the evolution of the latent vectors 
        for each data point in the dataset (if provided).
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_weight_decay)
    criterion = nn.MSELoss()

    model = model.to('cuda')

    history = []
    embeddings = []
    if len(dataset) > 0:
        embeddings = [ get_latent_vectors(dataset=dataset, model=model) ]
    for epoch in range(num_epochs):
        running_loss = 0.
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader, 0):
                inputs, _ = data
                inputs = inputs.to('cuda')

                loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs)
                running_loss += loss

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        history.append(running_loss/len(train_loader))

        if len(dataset) > 0:
            embeddings.append( get_latent_vectors(dataset=dataset, model=model) )

    embeddings = np.array(embeddings)

    return history, embeddings


def predict(image, model):
    '''
    Returns the output of model(image), and reshapes it to be compatible with plotting funtions such as plt.imshow().

    Args:
        image (3D numpy array): sample image with shape (n_channels, n_pixels_height, n_pixels_width).
        model (Pytorch Module): convolutional autoencoder that is prepared to process images such as 'image'.

    Returns:
        output_img (3D numpy array): output image with shape (n_pixels_height, n_pixels_width, n_channels)
    '''
    if image.shape[-1] <= 4:
        image = np.transpose(image, (2,0,1))
    n_channels, n_pixels_height, n_pixels_width = image.shape
    image = np.reshape(image, (1, n_channels, n_pixels_height, n_pixels_width))
    image = torch.from_numpy(image).float().to(next(model.parameters()).device)
    output_img = model(image)[0].detach().cpu().numpy()
    output_img = np.reshape(output_img, (n_channels, n_pixels_height, n_pixels_width))
    output_img = np.transpose(output_img, (1,2,0))
    return output_img


def get_latent_vectors(dataset, model, batch_size=128):
    '''
    Returns the latent activation vectors of the autoencoder model after passing all the images in the dataset.

    Args:
        dataset (numpy array): image dataset with shape 
        model (Pytorch Module): convolutional autoencoder that is prepared to process the images in dataset.

    Returns:
        latent_vectors (2D numpy array): latent activation vectors, matrix with shape (n_samples, n_hidden), where n_hidden is the number of units in the hidden layer.
    '''
    if dataset.shape[-1] <= 4:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            latent = model(inputs.to('cuda'))[1]
            latent_vectors.append(latent.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors)
    return latent_vectors


def find_max_activation_images(model, n_hidden, img_shape=[1, 84, 49]):
    images = []
    for i in range(n_hidden):
        # Initialize input image
        x = torch.randn(1, img_shape[0], img_shape[1], img_shape[2], device='cuda', requires_grad=True)

        # Use optimizer to perform gradient ascent
        optimizer = optim.Adam([x], lr=1e-3)

        for j in range(1000):
            optimizer.zero_grad()
            _, mu, _ = model(x)
            loss = -mu[0, i]  # maximize activation of ith unit
            loss.backward()
            optimizer.step()

        # Add image to list
        images.append(x.detach().cpu().numpy()[0, 0])

    return np.array(images)


def shuffle_2D_matrix(m):
    '''
    Shuffles a matrix across both axis (not only the first axis like numpy.permutation() or random.shuffle()).

    Args:
        m (2D numpy array): 2D matrix with arbitrary values.

    Returns:
        m_shuffled (2D numpy array): the original matrix 'm', with all the elements shuffled randomly.
    '''
    N = m.size
    ind_shuffled = np.arange(N)
    shuffle(ind_shuffled)
    ind_shuffled = ind_shuffled.reshape((m.shape[0], m.shape[1]))
    ind_x = (ind_shuffled/m.shape[1]).astype(np.int_)
    ind_y = (ind_shuffled%m.shape[1]).astype(np.int_)
    m_shuffled = m[ind_x, ind_y]
    return m_shuffled


def linear_decoding_score(embeddings, features, n_baseline=10000):
    '''
    Computes the score of linear regression of embeddings --> features. Features will normally be position (x,y) 
    or orientation (radians or in vectorial form).

    Args:
        embeddings (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        features (2D numpy array): 2D matrix containing the dependent variable, with shape (n_samples, n_features).
        n_baseline (int; default=10000): number of permutation tests (i.e., shuffling the embeddings matrix) to compute the baseline.

    Returns:
        scores (float list): a list with two scores: (1) the evaluation of the linear regression, and (2) an average & std 
                             of n_baseline random permutation tests.
    '''
    linear_model = LinearRegression()
    linear_model.fit(embeddings, features)
    linear_score = linear_model.score(embeddings, features)

    baselines = []
    for i in range(n_baseline):
        embeddings_shuffled = shuffle_2D_matrix(np.copy(embeddings))
        linear_model_baseline = LinearRegression()
        linear_model_baseline.fit(embeddings_shuffled, features)
        random_score = linear_model_baseline.score(embeddings_shuffled, features)
        baselines.append(random_score)

    baseline_score = [np.mean(baselines), np.std(baselines)]
    ratio = linear_score/(baseline_score[0])
    return linear_score, baseline_score, ratio


def linear_decoding_error(embeddings, features, norm=1):
    '''
    Computes the expected error of a linear decoder that uses the embeddings to predicts features (e.g. position in (x,y)).

    Args:
        embeddings (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        features (2D numpy array): 2D matrix containing the dependent variable, with shape (n_samples, n_features).
        norm (float; default=1): value used to normalize the MSE and bring it to a more convenient scale.

    Returns:
        mean_dist (float): average euclidean distance between the predictions of the decoder and the actual features, normalized by a scalar.
    '''
    linear_model = LinearRegression()
    linear_model.fit(embeddings, features)
    pred = linear_model.predict(embeddings)

    dist = np.sqrt(np.sum((pred - features)**2, axis=1))
    mean_dist = np.mean(dist) / norm

    return mean_dist

