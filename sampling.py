import random
import numpy as np

from distances import dtw
from utils import logger


def maximin_sampling(trajectories, sample_size, random_seed=42):
    """
    Sélectionne un échantillon représentatif.

    Principe :
    - on prend une trajectoire au hasard
    - ensuite on choisit à chaque fois la trajectoire la plus éloignée
      des trajectoires déjà sélectionnées

    C'est inspiré de l'idée de maximin sampling dans l'article.
    """

    random.seed(random_seed)

    n = len(trajectories)

    if sample_size >= n:
        return list(range(n))

    logger.info("Démarrage du maximin sampling")

    first_index = random.randint(0, n - 1)
    selected = [first_index]

    remaining = set(range(n))
    remaining.remove(first_index)

    min_distances = np.full(n, np.inf)

    for step in range(1, sample_size):
        last_selected = selected[-1]

        logger.info(f"Maximin sampling : {step}/{sample_size}")

        for idx in list(remaining):
            d = dtw(trajectories[idx], trajectories[last_selected])

            if d < min_distances[idx]:
                min_distances[idx] = d

        next_index = max(remaining, key=lambda idx: min_distances[idx])

        selected.append(next_index)
        remaining.remove(next_index)

    logger.info("Maximin sampling terminé")

    return selected