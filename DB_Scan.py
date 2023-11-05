from sklearn.cluster import DBSCAN
import numpy as np

def db_scan(eps, min_samples, data):

    dbscan = DBSCAN(eps=eps,min_samples=min_samples)
    labels = dbscan.fit_predict(data)

    # Create a mask for core points (labels != -1)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Get unique colors for each cluster
    unique_labels = set(labels)
    cluster_centroids =[]


    # Calculate the centroid for each cluster using core samples
    for label in unique_labels:
        cluster_points = data[core_samples_mask & (labels == label)]
        
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            cluster_centroids.append(centroid)
    return cluster_centroids