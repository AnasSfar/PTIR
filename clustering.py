import time
import numpy as np
from sklearn.cluster import DBSCAN

from config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES
from distances import dtw
from utils import logger


def compute_distance_matrix(sample):
    """
    Calcule la matrice DTW uniquement sur l'échantillon représentatif.
    """

    n = len(sample)
    dist_matrix = np.zeros((n, n))

    total_pairs = n * (n - 1) // 2
    computed_pairs = 0

    logger.info(f"Calcul DTW sur l'échantillon : {total_pairs} paires")

    start = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw(sample[i], sample[j])

            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

            computed_pairs += 1

        if i % 10 == 0 or i == n - 1:
            progress = (computed_pairs / total_pairs) * 100 if total_pairs else 100
            logger.info(
                f"DTW sample : {i + 1}/{n} | "
                f"{computed_pairs}/{total_pairs} paires | "
                f"{progress:.2f}%"
            )

    logger.info(f"Matrice DTW calculée en {time.time() - start:.2f}s")

    return dist_matrix


def run_dbscan(dist_matrix):
    """
    Lance DBSCAN sur la matrice de distance.
    """

    logger.info("Lancement de DBSCAN sur l'échantillon")

    dbscan = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="precomputed"
    )

    labels = dbscan.fit_predict(dist_matrix)

    return labels


def summarize_clusters(labels):
    clusters = set(labels)
    nb_clusters = len(clusters) - (1 if -1 in clusters else 0)
    nb_noise = list(labels).count(-1)

    logger.info(f"Labels trouvés : {clusters}")
    logger.info(f"Nombre de clusters : {nb_clusters}")
    logger.info(f"Bruit : {nb_noise}")