import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def compute_distance_matrix(arr_list):
    """
    Given a list of 2D arrays, compute the pairwise distance matrix using Pearson correlation.
    """
    # Get the number of arrays in the list
    n_arrays = len(arr_list)
    # Create an empty distance matrix with shape (n_arrays, n_arrays)
    dist_mat = np.zeros((n_arrays, n_arrays))
    # Compute the pairwise distance between all pairs of arrays using Pearson correlation
    for i in range(n_arrays):
        for j in range(i+1, n_arrays):
            corr = np.corrcoef(arr_list[i], arr_list[j])[0, 1]
            dist = 1 - np.abs(corr)
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    # Return the distance matrix
    return dist_mat

def cluster_and_plot(arr_list, kind='betas'):
    """
    Given a list of 2D arrays, compute the pairwise distance matrix, cluster it using k-means,
    and plot the t-SNE projection of the distance matrix, colored by the cluster labels.
    """
    # Compute the distance matrix using Pearson correlation
    dist_mat = compute_distance_matrix(arr_list)
    # Set the number of clusters for k-means
    n_clusters = 10
    # Apply k-means clustering to the distance matrix
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(dist_mat)
    # Compute the t-SNE projection of the distance matrix
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb = tsne.fit_transform(dist_mat)
    # Plot the t-SNE projection, coloring each point by its cluster label
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(emb[:,0], emb[:,1], c=labels, cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Clusters")
    ax.add_artist(legend)
    ax.set_title(f"t-SNE projection of {kind} distance matrix")
    plt.show()
