from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch


def k_means_clustering(data, n_clusters, random_state=42, n_init=10):
    data_np = data.cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit(data_np)

    return torch.from_numpy(kmeans.labels_)


def hierarchical_clustering(data, n_clusters):
    data_np = data.cpu().numpy()

    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data_np)

    return torch.from_numpy(agg_clustering.labels_)


def visualize_2d(noisy_tensors, centroids, labels):
    pca = PCA(n_components=2)
    noisy_tensors_2d = pca.fit_transform(noisy_tensors)

    plt.scatter(noisy_tensors_2d[:, 0], noisy_tensors_2d[:, 1], c=labels, cmap='viridis')

    if centroids is not None:
        centroids_2d = pca.transform(torch.stack(centroids))
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', s=100, alpha=0.5)

    plt.show()


def visualize_3d(noisy_tensors, centroids, labels):
    pca = PCA(n_components=3)
    noisy_tensors_3d = pca.fit_transform(noisy_tensors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(noisy_tensors_3d[:, 0], noisy_tensors_3d[:, 1], noisy_tensors_3d[:, 2], c=labels, cmap='viridis')

    if centroids is not None:
        centroids_3d = pca.transform(torch.stack(centroids))
        ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], c='red', s=100, alpha=0.5)

    plt.show()
