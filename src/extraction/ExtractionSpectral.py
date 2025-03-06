import logging
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from ExtractionBase import BaseExtraction

logging.basicConfig(level=logging.info)
log = logging.getLogger(__name__)


class Spectral_Clustering_Impl(BaseExtraction):
    @staticmethod
    def clustering(features, **kwargs):
        sqrt_n = int(np.sqrt(len(features)))
        features = [i[0] for i in features]
        k = sqrt_n
        
        best_ch_score = -1
        best_silhouette_score = 0
        best_gamma = 0.01
        best_k = k
        
        best_clusters = [0] * len(features)
        while k > 1:
            for index, gamma in enumerate((0.01,0.1,1,10)):
                clusters = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(features)
                ch_score = metrics.calinski_harabasz_score(features, clusters)
                avg_silhouette = metrics.silhouette_score(features, clusters)
                if ch_score > best_ch_score:
                    best_ch_score = ch_score
                    best_clusters = clusters.copy()
                    best_gamma = gamma
                    best_k = k
                if avg_silhouette > best_silhouette_score:
                    best_silhouette_score = avg_silhouette
            k -= 1
        
        best_centers, center_indices = Spectral_Clustering_Impl.__get_centers(best_clusters, **kwargs)
        return best_clusters, best_centers, best_k, center_indices
    
    
    @staticmethod
    def __get_centers(best_clusters, **kwargs):
        hybrid_features = kwargs.get('hybrid_features', None)
        image_features = kwargs.get('image_features', None)
        text_features = kwargs.get('text_features', None)
        center_indicies = []
        centers = []
        for cluster_id in np.unique(best_clusters):
            cluster_indices = np.where(best_clusters == cluster_id)[0]
            cluster_image_features = image_features[cluster_indices]
            cluster_image_features = cluster_image_features[:, 0, :]
            cluster_text_features = text_features
            
            def cluster_with_text():
                similarities = cdist(cluster_image_features, cluster_text_features, metric='cosine')[:, 0]
                top_indices = np.argsort(similarities)[:1]
                centers.extend(hybrid_features[cluster_indices[top_indices]])
                center_indicies.extend(cluster_indices[top_indices])
            
            def cluster_without_text():
                distances = cdist(cluster_image_features, cluster_image_features, metric='euclidean')
                min_idx = -1
                # Iterate over the values in the upper right corner of the matrix
                k = len(cluster_image_features)
                min_distances = np.empty((k))
                for i in range(k):
                    min_distances[i] = np.sum(distances[i, :])
                min_idx = np.argsort(min_distances)[:1]
                centers.extend(cluster_image_features[min_idx])
                center_indicies.extend(cluster_indices[min_idx])

            if text_features is not None:
                cluster_with_text()
            else:
                cluster_without_text()
        return centers, center_indicies
