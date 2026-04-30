import time
import pandas as pd

from config import GPS_FOLDER, SAMPLE_SIZE, RANDOM_SEED, OUTPUT_FILE
from data_loader import load_trajectories
from sampling import maximin_sampling
from clustering import compute_distance_matrix, run_dbscan, summarize_clusters
from propagation import propagate_labels
from visualization import plot_sample_clusters
from utils import logger


def main():
    total_start = time.time()

    logger.info("Démarrage du pipeline Fast-clusiVAT simplifié")

    # 1. Charger toutes les trajectoires
    trajectories, names = load_trajectories(GPS_FOLDER)

    if len(trajectories) == 0:
        logger.error("Aucune trajectoire valide trouvée.")
        return

    # 2. Sélectionner un échantillon représentatif
    sample_indices = maximin_sampling(
        trajectories,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )

    sample = [trajectories[i] for i in sample_indices]
    sample_names = [names[i] for i in sample_indices]

    logger.info(f"Échantillon représentatif sélectionné : {len(sample)} trajectoires")

    # 3. Calculer la matrice DTW seulement sur l'échantillon
    dist_matrix = compute_distance_matrix(sample)

    # 4. Clusteriser l'échantillon
    sample_labels = run_dbscan(dist_matrix)

    summarize_clusters(sample_labels)

    # 5. Propager les labels à tout le dataset
    final_labels = propagate_labels(
        all_trajectories=trajectories,
        sample_indices=sample_indices,
        sample_labels=sample_labels
    )

    # 6. Sauvegarder tous les résultats
    results = pd.DataFrame({
        "file": names,
        "cluster": final_labels,
        "is_sample": [i in sample_indices for i in range(len(names))]
    })

    results.to_csv(OUTPUT_FILE, index=False)

    logger.info(f"Résultats complets sauvegardés dans : {OUTPUT_FILE}")

    # 7. Visualiser l'échantillon
    plot_sample_clusters(sample, sample_labels)

    logger.info(f"Temps total : {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()