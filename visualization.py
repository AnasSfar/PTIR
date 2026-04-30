import matplotlib.pyplot as plt

from utils import logger


def plot_sample_clusters(sample, labels):
    """
    Affiche uniquement les trajectoires de l'échantillon.
    """

    logger.info("Affichage des clusters de l'échantillon")

    plt.figure(figsize=(10, 8))

    for traj, label in zip(sample, labels):
        lats = [p[0] for p in traj]
        lons = [p[1] for p in traj]

        if label == -1:
            plt.plot(lons, lats, color="gray", alpha=0.2)
        else:
            plt.plot(lons, lats, alpha=0.6)

    plt.title("Clusters sur l'échantillon représentatif")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()