import numpy as np


def point_distance(p1, p2):
    """
    Distance euclidienne simple entre deux points GPS.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def dtw(traj1, traj2):
    """
    DTW avec fenêtre de recherche pour accélérer.

    Cette version est une approximation plus rapide.
    """

    n, m = len(traj1), len(traj2)

    window = max(abs(n - m), min(n, m) // 2)

    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0

    for i in range(1, n + 1):
        start_j = max(1, i - window)
        end_j = min(m + 1, i + window + 1)

        for j in range(start_j, end_j):
            cost = point_distance(traj1[i - 1], traj2[j - 1])

            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1]
            )

    return dp[n, m]