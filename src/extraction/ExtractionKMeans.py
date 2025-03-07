import os
import logging
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from ExtractionBase import BaseExtraction

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class KMeans_Extraction_Impl(BaseExtraction):
    @staticmethod
    def clustering(features, **kwargs):
        # Calculate sqrt(n) and initialise k and clustering results
        sqrt_n = int(np.sqrt(len(features)))
        features = [i[0] for i in features]
        k = sqrt_n
        best_avg_silhouette = -1

        # Results of the selection of the initial center
        clusters, centers = KMeans_Extraction_Impl.__kmeans_init(features)
        
        # Remove duplicate centers
        unique_centers, indices = np.unique(centers, axis=0, return_index=True)
        centers = unique_centers
        clusters = np.array([indices[cluster] for cluster in clusters])
        k = len(centers)
        
        # init best records
        best_clusters = clusters.copy()
        best_centers = centers.copy()
        best_k = k
        center_indices = []
        for center in best_centers:
            center_index = np.where((features == center).all(axis=1))[0][0]
            center_indices.append(center_index)
        
        # Iterative Procedure
        while k > 2:
            # Calculate the Euclidean distance between cluster centers
            cluster_centers = centers
            distances = cdist(cluster_centers, cluster_centers, metric='euclidean')

            # Find the indexes of the two the nearest clusters
            min_distance = np.inf
            merge_cluster_indices = None
            
            # Iterate over the values in the upper right corner of the matrix
            for i in range(k):
                for j in range(i + 1, k):
                    if distances[i, j] < min_distance:
                        min_distance = distances[i, j]
                        merge_cluster_indices = (i, j)

            # Merge the two the nearest clusters and change the high cluster number to the low cluster number
            merged_cluster = np.where(clusters == merge_cluster_indices[1], merge_cluster_indices[0], clusters)

            # Update clustering results
            clusters = np.where(merged_cluster > merge_cluster_indices[1], merged_cluster - 1, merged_cluster)

            # Update the cluster center, selecting the actual data point as the new cluster center
            new_centers = []
            for cluster_id in range(k - 1):
                # Get samples of the current cluster
                cluster_samples = [features[i] for i in range(len(features)) if clusters[i] == cluster_id]
                if len(cluster_samples) == 0:
                    continue
                # Calculate the current cluster mean
                cluster_mean = np.mean(cluster_samples, axis=0)
                # Calculate the Euclidean distance between the sample and the centre point to find the actual center
                distances = np.linalg.norm(cluster_samples - cluster_mean, axis=1)
                closest_sample_index = np.argmin(distances)
                # Choose the nearest sample as the new cluster centroid
                new_centers.append(cluster_samples[closest_sample_index])
            centers = new_centers
            
            # update number of cluster
            k -= 1
            
            # Calculate Silhouette Coefficient and update the best records
            avg_silhouette = silhouette_score(features, clusters)
            log.info(f"avg_silhouette: {avg_silhouette}")
            if avg_silhouette > best_avg_silhouette:
                best_avg_silhouette = avg_silhouette
                best_k = k
                best_clusters = clusters.copy()
                best_centers = centers.copy()
                center_indices = []
                for cluster_center in best_centers:
                    center_index = np.where((features == cluster_center).all(axis=1))[0][0]
                    center_indices.append(center_index)

        log.info(f"best_avg_sc: {best_avg_silhouette}")
        return best_clusters, best_centers, best_k, center_indices
    
    @staticmethod
    def __kmeans_init(data):
        # calculate sqrt(n)
        n = len(data)
        log.info("In the process of initialising the center")
        sqrt_n = int(np.sqrt(n))
        centers = []
        label = []

        # pick init_center
        while len(centers) < sqrt_n:
            log.info(f"In the process of initialising the center {len(centers)}")
            sse_min = float('inf')
            for i in range(n):
                center = centers.copy()
                def if_data_in_centers():
                    for cen in center:
                        if np.all(data[i] == cen):
                            return True
                    return False
                if not if_data_in_centers():
                    center.append(data[i])
                    center = np.array(center)
                    sse = 0.0

                    # Cluster operation
                    cluster_labels = np.zeros(len(data)).astype(int)
                    for k in range(len(data)):
                        distances = [np.sqrt(np.sum((data[k] - cen) ** 2)) for cen in center]
                        nearest_cluster = np.argmin(distances)
                        cluster_labels[k] = nearest_cluster

                    # Based on the results of the cluster operation,calculate sse
                    for j in range(len(center)):
                        # Get the data points of the jth cluster
                        cluster_points = []
                        for l in range(len(cluster_labels)):
                            if cluster_labels[l] == j:
                                cluster_points.append(data[l])
                        singe_sse = 0.0
                        for point in cluster_points:
                            squared_errors = np.linalg.norm(point - center[j])
                            singe_sse += squared_errors
                        sse += singe_sse

                    if sse < sse_min:
                        sse_min = sse
                        join_center = data[i]
                        label = cluster_labels.copy()
            centers.append(join_center)

        return np.array(label), np.array(centers)
    
    @staticmethod
    def __merge_two_nodes(features, centers):
        # get the best centers when k = 2
        distances = cdist(centers, features, metric='euclidean')
        min_distance = np.inf
        for i in range(len(centers)):
            distance = np.sum(distances[i])
            if distance < min_distance:
                min_distance = distance
                min_index = i
        return centers[min_index]