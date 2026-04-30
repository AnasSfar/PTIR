import time

from distances import dtw
from utils import logger


def propagate_labels(
    all_trajectories,
    sample_indices,
    sample_labels
):
    """
    Propage les labels de l'échantillon à toutes les autres trajectoires.

    Pour chaque trajectoire non sélectionnée :
    - on la compare aux trajectoires de l'échantillon
    - on récupère le label du plus proche voisin

    C'est une version simplifiée du nearest prototype rule.
    """

    logger.info("Démarrage de la propagation des labels")

    start = time.time()

    sample_index_to_label = {
        idx: label for idx, label in zip(sample_indices, sample_labels)
    }

    final_labels = [-999] * len(all_trajectories)

    # D'abord, on place les labels connus de l'échantillon
    for idx, label in sample_index_to_label.items():
        final_labels[idx] = label

    total = len(all_trajectories)

    for idx, traj in enumerate(all_trajectories):
        if final_labels[idx] != -999:
            continue

        best_distance = float("inf")
        best_label = -1

        for sample_idx, sample_label in zip(sample_indices, sample_labels):
            # On évite de propager depuis un bruit si possible
            if sample_label == -1:
                continue

            d = dtw(traj, all_trajectories[sample_idx])

            if d < best_distance:
                best_distance = d
                best_label = sample_label

        final_labels[idx] = best_label

        if idx % 100 == 0:
            logger.info(f"Propagation : {idx}/{total} trajectoires traitées")

    logger.info(f"Propagation terminée en {time.time() - start:.2f}s")

    return final_labels