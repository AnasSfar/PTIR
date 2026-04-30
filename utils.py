import logging
import numpy as np

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)

logger = setup_logger()


def downsample_trajectory(traj, max_points):
    """
    Réduit une trajectoire trop longue à max_points points.
    """
    if len(traj) <= max_points:
        return traj

    indices = np.linspace(0, len(traj) - 1, max_points).astype(int)
    return [traj[i] for i in indices]