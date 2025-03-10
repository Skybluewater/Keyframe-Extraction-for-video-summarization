# by @https://github.com/ttharden/Keyframe-extraction
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def kmeans_init(data):
    log.info("In the process of initialising the center")
    n = len(data)
    # calculate sqrt(n)
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
                # print(center)
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
