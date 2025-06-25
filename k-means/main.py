import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score


def euclidian_distance(centroid, data_point):
    return ((centroid[0] - data_point[0]) ** 2 +
            (centroid[1] - data_point[1]) ** 2 +
            (centroid[2] - data_point[2]) ** 2 +
            (centroid[3] - data_point[3]) ** 2) ** 0.5


def expectation(centroids, data_point, k):
    distance_min = np.inf
    nearest_centroid = None
    for i in range(0, k):
        distance_to_centroid = euclidian_distance(centroids[i], data_point)
        if distance_to_centroid < distance_min:
            distance_min = distance_to_centroid
            nearest_centroid = i
    return nearest_centroid


def maximization(old_centroids, clusters, k):
    new_centroids = []
    for i in range(0, k):
        centroid_sum = [0, 0, 0, 0]
        if len(clusters[i]) != 0:
            for data_point in clusters[i]:
                for j, pos in enumerate(data_point):
                    centroid_sum[j] += pos
            new_centroid = [pos / len(clusters[i]) for pos in centroid_sum]
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(old_centroids[i])
    return new_centroids


def readIris(name_file):
    data_iris = pd.read_csv(name_file)
    columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    return data_iris[columns].values.tolist()


def initializeCentroids(data, k):
    centroids = []
    if k == 3:
        centroids.append(data[0]), centroids.append(data[50]), centroids.append(data[100])

    if k == 5:
        centroids.append(data[0]), centroids.append(data[30]), centroids.append(data[60])
        centroids.append(data[90]), centroids.append(data[120])

    return centroids


def initializeClusters(k):
    cluster = []
    for i in range(0, k):
        cluster.append(list())
    return cluster


def plot_clusters(clusters, centroids, k):
    plt.clf()
    for i in range(k):
        cluster = np.array(clusters[i])
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')

    centroids_array = np.array(centroids)
    plt.scatter(centroids_array[:, 0], centroids_array[:, 1], marker='x', color='black', label='Centroids')
    plt.xlabel('Sepal LengthCm')
    plt.ylabel('Sepal WidthCm')
    plt.title(f'K-mean Iris dataset k{k}')
    plt.legend()
    plt.savefig(f'hardcode-k{k}')


def calculate_silhouette(clusters, data_points):
    labels = np.zeros(len(data_points), dtype=int)
    for i, cluster in enumerate(clusters):
        labels[[data_points.index(point) for point in cluster]] = i
    return silhouette_score(data_points, labels)


def k_means(data_points, k):
    centroids = initializeCentroids(data_points, k)
    clusters = None
    for iterations in range(0, 10):
        clusters = initializeClusters(k)

        for i in range(len(data_points)):
            nearest_cluster = expectation(centroids, data_points[i], k)
            clusters[nearest_cluster].append(data_points[i])

        new_centroids = maximization(centroids, clusters, k)
        centroids = new_centroids
    silhouette = calculate_silhouette(clusters, data_points)
    print(f"Silhouette Score para k={k} clusters: {silhouette}")
    #   plot_clusters(clusters, centroids, k)


def run():
    data_points = readIris('iris.csv')
    k_means(data_points, k=3)
    k_means(data_points, k=5)


if __name__ == '__main__':
    begin_time = time.time()
    run()
    end_time = time.time()
    print(f'tempo de execução K-means hardcode: {end_time - begin_time} segundos')
