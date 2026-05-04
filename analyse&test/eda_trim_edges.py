import os
import glob
import random
import logging
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = r"C:\Users\sfara\Documents\GitHub\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"

OUT_DIR = "gps_trim_eda_outputs"
LOG_FILE = "gps_trim_eda.log"

SAMPLE_SIZE = 50
RANDOM_SEED = 42
MAX_POINTS_PER_FILE = 10000

SPEED_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
RADIUS_FACTORS = [0.02, 0.05, 0.10]
MIN_RADIUS = 50
MAX_RADIUS = 300

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def load_trajectory(file):
    df = pd.read_csv(file)

    required = {"LATITUDE", "LONGITUDE", "UTC_DATE", "UTC_TIME"}
    if not required.issubset(df.columns):
        return None

    df = df.dropna(subset=["LATITUDE", "LONGITUDE", "UTC_DATE", "UTC_TIME"])

    df = df[
        (df["LATITUDE"].between(-90, 90)) &
        (df["LONGITUDE"].between(-180, 180))
    ]

    df["DATETIME"] = pd.to_datetime(
        df["UTC_DATE"].astype(str) + " " + df["UTC_TIME"].astype(str),
        errors="coerce"
    )

    df = df.dropna(subset=["DATETIME"])
    df = df.sort_values("DATETIME").reset_index(drop=True)

    if len(df) < 10:
        return None

    if len(df) > MAX_POINTS_PER_FILE:
        step = int(np.ceil(len(df) / MAX_POINTS_PER_FILE))
        df = df.iloc[::step].copy().reset_index(drop=True)

    df["delta_t"] = df["DATETIME"].diff().dt.total_seconds()

    distances = [np.nan]
    for i in range(1, len(df)):
        distances.append(
            haversine(
                df.loc[i - 1, "LATITUDE"],
                df.loc[i - 1, "LONGITUDE"],
                df.loc[i, "LATITUDE"],
                df.loc[i, "LONGITUDE"]
            )
        )

    df["dist_m"] = distances
    df["speed_mps"] = df["dist_m"] / df["delta_t"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["delta_t", "dist_m", "speed_mps"])
    df = df[df["delta_t"] > 0]

    if len(df) < 10:
        return None

    return df.reset_index(drop=True)


def compute_trim(df, speed_threshold, radius_factor):
    total_distance = df["dist_m"].sum()

    radius = min(
        MAX_RADIUS,
        max(MIN_RADIUS, total_distance * radius_factor)
    )

    start_lat = df.loc[0, "LATITUDE"]
    start_lon = df.loc[0, "LONGITUDE"]
    end_lat = df.loc[len(df) - 1, "LATITUDE"]
    end_lon = df.loc[len(df) - 1, "LONGITUDE"]

    start_cut = 0

    for i in range(len(df)):
        dist_from_start = haversine(
            start_lat,
            start_lon,
            df.loc[i, "LATITUDE"],
            df.loc[i, "LONGITUDE"]
        )

        near_start = dist_from_start <= radius
        slow = df.loc[i, "speed_mps"] <= speed_threshold

        if near_start and slow:
            start_cut = i
        else:
            break

    end_cut = len(df) - 1

    for i in range(len(df) - 1, -1, -1):
        dist_from_end = haversine(
            end_lat,
            end_lon,
            df.loc[i, "LATITUDE"],
            df.loc[i, "LONGITUDE"]
        )

        near_end = dist_from_end <= radius
        slow = df.loc[i, "speed_mps"] <= speed_threshold

        if near_end and slow:
            end_cut = i
        else:
            break

    start_removed_points = start_cut + 1
    end_removed_points = len(df) - end_cut
    total_removed_points = start_removed_points + end_removed_points

    start_removed_distance = df.iloc[:start_removed_points]["dist_m"].sum()
    end_removed_distance = df.iloc[end_cut:]["dist_m"].sum()

    kept_points = max(0, len(df) - total_removed_points)

    return {
        "speed_threshold": speed_threshold,
        "radius_factor": radius_factor,
        "radius_m": radius,
        "total_distance_m": total_distance,
        "total_points": len(df),
        "start_removed_points": start_removed_points,
        "end_removed_points": end_removed_points,
        "total_removed_points": total_removed_points,
        "removed_ratio": total_removed_points / len(df),
        "kept_points": kept_points,
        "start_removed_distance_m": start_removed_distance,
        "end_removed_distance_m": end_removed_distance,
        "total_removed_distance_m": start_removed_distance + end_removed_distance,
        "removed_distance_ratio": (start_removed_distance + end_removed_distance) / total_distance if total_distance > 0 else 0
    }


def save_boxplot(results, metric, filename, ylabel):
    df = pd.DataFrame(results)

    labels = []
    data = []

    for speed in SPEED_THRESHOLDS:
        for factor in RADIUS_FACTORS:
            subset = df[
                (df["speed_threshold"] == speed) &
                (df["radius_factor"] == factor)
            ]

            if len(subset) > 0:
                labels.append(f"v={speed}\nr={int(factor*100)}%")
                data.append(subset[metric])

    plt.figure(figsize=(13, 6))
    plt.boxplot(data, labels=labels)
    plt.ylabel(ylabel)
    plt.title(metric)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=180)
    plt.close()


def save_cdf(data, title, xlabel, filename):
    s = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna()

    if len(s) == 0:
        return

    x = np.sort(s)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative probability")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=180)
    plt.close()


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    random.seed(RANDOM_SEED)
    files = random.sample(files, min(SAMPLE_SIZE, len(files)))

    logger.info(f"CSV analyzed: {len(files)}")

    results = []

    for idx, file in enumerate(files, start=1):
        logger.info(f"[{idx}/{len(files)}] {file}")

        try:
            df = load_trajectory(file)

            if df is None:
                continue

            for speed in SPEED_THRESHOLDS:
                for factor in RADIUS_FACTORS:
                    result = compute_trim(df, speed, factor)
                    result["file"] = os.path.basename(file)
                    results.append(result)

        except Exception as e:
            logger.error(f"Error: {file} - {e}")

    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUT_DIR, "trim_experiments_results.csv")
    results_df.to_csv(results_path, index=False)

    save_boxplot(
        results,
        "removed_ratio",
        "01_removed_points_ratio_boxplot.png",
        "Removed points ratio"
    )

    save_boxplot(
        results,
        "removed_distance_ratio",
        "02_removed_distance_ratio_boxplot.png",
        "Removed distance ratio"
    )

    save_boxplot(
        results,
        "total_removed_distance_m",
        "03_removed_distance_m_boxplot.png",
        "Removed distance (m)"
    )

    save_boxplot(
        results,
        "radius_m",
        "04_adaptive_radius_boxplot.png",
        "Adaptive radius (m)"
    )

    best_subset = results_df[
        (results_df["speed_threshold"] == 1.0) &
        (results_df["radius_factor"] == 0.05)
    ]

    save_cdf(
        best_subset["removed_ratio"],
        "CDF of removed point ratio, v=1m/s, radius=5%",
        "Removed points ratio",
        "05_removed_ratio_cdf_selected.png"
    )

    save_cdf(
        best_subset["removed_distance_ratio"],
        "CDF of removed distance ratio, v=1m/s, radius=5%",
        "Removed distance ratio",
        "06_removed_distance_ratio_cdf_selected.png"
    )

    summary_path = os.path.join(OUT_DIR, "trim_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("TRIM EDGE EDA SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("Goal:\n")
        f.write("Test different rules for removing local origin/destination parts.\n\n")

        f.write("Tested speed thresholds:\n")
        f.write(str(SPEED_THRESHOLDS) + "\n\n")

        f.write("Tested radius factors:\n")
        f.write(str(RADIUS_FACTORS) + "\n\n")

        f.write("Recommended default to inspect:\n")
        f.write("speed_threshold = 1.0 m/s\n")
        f.write("radius_factor = 0.05\n")
        f.write("radius = min(300m, max(50m, 5% of trajectory length))\n\n")

        f.write("Global statistics:\n")
        f.write(str(results_df.describe()))
        f.write("\n\n")

        f.write("Average by configuration:\n")
        f.write(
            str(
                results_df
                .groupby(["speed_threshold", "radius_factor"])
                [["removed_ratio", "removed_distance_ratio", "total_removed_distance_m", "radius_m"]]
                .mean()
            )
        )

    logger.info(f"Results saved in: {os.path.abspath(OUT_DIR)}")
    logger.info(f"CSV results: {results_path}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()