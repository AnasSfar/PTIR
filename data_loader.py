import os
import pandas as pd

from config import MIN_POINTS, MAX_POINTS
from utils import logger, downsample_trajectory


def load_trajectories(folder):
    """
    Charge toutes les trajectoires GPS.

    Chaque fichier CSV = une trajectoire.
    On garde seulement LATITUDE et LONGITUDE.
    """

    logger.info(f"Chargement des trajectoires depuis : {folder}")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dossier introuvable : {folder}")

    trajectories = []
    names = []

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    logger.info(f"Nombre de fichiers CSV trouvés : {len(files)}")

    for index, file in enumerate(files, start=1):
        path = os.path.join(folder, file)

        try:
            df = pd.read_csv(path)

            if not {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
                logger.warning(f"Colonnes manquantes dans : {file}")
                continue

            df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

            traj = list(zip(df["LATITUDE"], df["LONGITUDE"]))

            if len(traj) >= MIN_POINTS:
                traj = downsample_trajectory(traj, MAX_POINTS)
                trajectories.append(traj)
                names.append(file)

        except Exception as e:
            logger.warning(f"Erreur lecture {file} : {e}")

        if index % 500 == 0:
            logger.info(f"Chargement : {index}/{len(files)} fichiers traités")

    logger.info(f"Trajectoires valides chargées : {len(trajectories)}")

    return trajectories, names