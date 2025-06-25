import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def associateLabels(labels, data_point, k):
    clusters = []
    for i in range(0, k):
        clusters.append(list())
    for i in range(len(data_point)):
        clusters[labels[i]].append(data_point[i])
    return clusters


def plot_PCA(clusters, centroids, k, number_components):
    plt.clf()
    if number_components == 1:
        for i in range(k):
            cluster = np.array(clusters[i])
            plt.scatter(cluster[:, 0], np.zeros_like(cluster[:, 0]), label=f'Cluster {i + 1}')
        centroids_array = np.array(centroids)
        plt.scatter(centroids_array[:, 0], np.zeros_like(centroids_array[:, 0]), marker='x', color='black', label='Centroides')
    else:
        for i in range(k):
            cluster = np.array(clusters[i])
            plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')
        centroids_array = np.array(centroids)
        plt.scatter(centroids_array[:, 0], centroids_array[:, 1], marker='x', color='black', label='Centroides')

    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title(f'K-means Iris-dataset k={k} e número de componentes={number_components}')
    plt.legend()
    plt.savefig(f'sklearn-k{k}-components{number_components}')


def readIris(name_file):
    data_iris = pd.read_csv(name_file)
    columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    return data_iris[columns].values.tolist()


def k_means(data_points, k, number_components=4):
    kmeans = KMeans(n_clusters=k, init='random', max_iter=10)
    kmeans.fit(data_points)
    silhouette = silhouette_score(data_points, kmeans.labels_)
    print(f"Silhouette Score para k={k} clusters: {silhouette}")

    if number_components != 4:
        clusters = associateLabels(kmeans.labels_, data_points, k)
        plot_PCA(clusters, kmeans.cluster_centers_, k, number_components)


def run():
    begin_time = time.time()
    data_points = readIris('iris.csv')
    k_means(data_points, k=3)
    k_means(data_points, k=5)
    end_time = time.time()
    print(f'tempo de execução K-means sklearn {end_time - begin_time} segundos')

    #   Nos testes realizados k = 3 apresenta um melhor Silhouette Score quando comparado com k = 5 ou então quando
    #   comparado com o algoritmo hardcode
    #   Então vamos realizar o PCA para k = 3

    k = 3
    number_components = 1
    pca = PCA(number_components)
    k_means(pca.fit_transform(data_points), k, number_components)
    del pca

    number_components = 2
    pca = PCA(number_components)
    k_means(pca.fit_transform(data_points), k, number_components)


if __name__ == '__main__':
    run()
