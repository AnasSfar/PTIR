import os
import glob
import random
import logging
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

DATA_DIR = r"C:\Users\sfara\Documents\GitHub\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"

OUT_DIR = "gps_decision_graphs"
LOG_FILE = "gps_decision_graphs.log"

SAMPLE_SIZE = 50
RANDOM_SEED = 42
MAX_POINTS_PER_FILE_FOR_EDA = 10000

MIN_STOP_SPEED = 1.0  # m/s

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


# =========================
# UTILS
# =========================

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


def clean_series(data, positive_only=False, max_value=None):
    s = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna()

    if positive_only:
        s = s[s > 0]

    if max_value is not None:
        s = s[s <= max_value]

    return s


def save_cdf(data, title, xlabel, filename, thresholds=None, log_x=False, max_value=None):
    s = clean_series(data, positive_only=log_x, max_value=max_value)

    if len(s) == 0:
        logger.warning(f"No data for {filename}")
        return

    x = np.sort(s)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(11, 6))
    plt.plot(x, y, linewidth=2)

    if log_x:
        plt.xscale("log")

    if thresholds:
        for t in thresholds:
            pct = (s <= t).mean()
            plt.axvline(t, linestyle="--", alpha=0.8)
            plt.text(
                t,
                0.08,
                f"{t:g}\n{pct:.2%}",
                rotation=90,
                verticalalignment="bottom"
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=180)
    plt.close()

    logger.info(f"Saved: {path}")


def save_hist(data, title, xlabel, filename, max_value=None):
    s = clean_series(data, max_value=max_value)

    if len(s) == 0:
        logger.warning(f"No data for {filename}")
        return

    plt.figure(figsize=(11, 6))
    plt.hist(s, bins=80)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=180)
    plt.close()

    logger.info(f"Saved: {path}")


def add_stop_durations(df, stop_durations):
    """
    Detect consecutive low-speed periods and compute their duration.
    A staypoint candidate = consecutive points with speed < MIN_STOP_SPEED.
    """

    is_stop = df["speed"] < MIN_STOP_SPEED
    start_time = None
    last_time = None

    for i in range(len(df)):
        current_time = df.iloc[i]["DATETIME"]

        if is_stop.iloc[i]:
            if start_time is None:
                start_time = current_time
            last_time = current_time
        else:
            if start_time is not None and last_time is not None:
                duration_min = (last_time - start_time).total_seconds() / 60
                if duration_min > 0:
                    stop_durations.append(duration_min)

            start_time = None
            last_time = None

    if start_time is not None and last_time is not None:
        duration_min = (last_time - start_time).total_seconds() / 60
        if duration_min > 0:
            stop_durations.append(duration_min)


def save_summary(all_delta_t, all_dist, all_speed, stop_durations):
    summary_path = os.path.join(OUT_DIR, "summary_decision_stats.txt")

    sections = {
        "DELTA TIME SECONDS": all_delta_t,
        "DISTANCE METERS": all_dist,
        "SPEED MPS": all_speed,
        "STOP DURATIONS MINUTES": stop_durations,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GPS DECISION EDA SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        for name, data in sections.items():
            s = clean_series(data)

            f.write("\n" + "=" * 70 + "\n")
            f.write(name + "\n")
            f.write("=" * 70 + "\n")

            if len(s) == 0:
                f.write("No data\n")
                continue

            f.write(str(s.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])))
            f.write("\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("THRESHOLD COVERAGE\n")
        f.write("=" * 70 + "\n")

        delta_s = clean_series(all_delta_t)
        dist_s = clean_series(all_dist)
        speed_s = clean_series(all_speed)
        stop_s = clean_series(stop_durations)

        f.write("\nDelta time thresholds:\n")
        for t in [300, 600, 900, 1800]:
            f.write(f"<= {t}s: {(delta_s <= t).mean():.4f}\n")

        f.write("\nDistance thresholds:\n")
        for d in [50, 100, 200, 500]:
            f.write(f"<= {d}m: {(dist_s <= d).mean():.4f}\n")

        f.write("\nSpeed thresholds:\n")
        for v in [1, 5, 15, 30]:
            f.write(f"<= {v}m/s: {(speed_s <= v).mean():.4f}\n")

        f.write("\nStaypoint duration thresholds:\n")
        for m in [5, 8, 10, 15, 20, 180]:
            f.write(f"<= {m}min: {(stop_s <= m).mean():.4f}\n")

    logger.info(f"Summary saved: {summary_path}")


# =========================
# MAIN
# =========================

def main():
    logger.info("Starting GPS decision EDA")

    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    logger.info(f"CSV found: {len(all_files)}")

    if not all_files:
        logger.error("No CSV files found.")
        return

    random.seed(RANDOM_SEED)
    files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))

    all_delta_t = []
    all_dist = []
    all_speed = []
    stop_durations = []

    required = {"LATITUDE", "LONGITUDE", "UTC_DATE", "UTC_TIME"}

    for idx, file in enumerate(files, start=1):
        logger.info(f"[{idx}/{len(files)}] Reading {file}")

        try:
            df = pd.read_csv(file)

            if not required.issubset(df.columns):
                logger.warning(f"Missing columns, skipped: {file}")
                continue

            df = df.dropna(subset=["LATITUDE", "LONGITUDE", "UTC_DATE", "UTC_TIME"])

            df = df[
                (df["LATITUDE"].between(-90, 90))
                & (df["LONGITUDE"].between(-180, 180))
            ]

            df["DATETIME"] = pd.to_datetime(
                df["UTC_DATE"].astype(str) + " " + df["UTC_TIME"].astype(str),
                errors="coerce"
            )

            df = df.dropna(subset=["DATETIME"])
            df = df.sort_values("DATETIME").reset_index(drop=True)

            if len(df) < 2:
                continue

            if len(df) > MAX_POINTS_PER_FILE_FOR_EDA:
                step = int(np.ceil(len(df) / MAX_POINTS_PER_FILE_FOR_EDA))
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

            df["dist"] = distances
            df["speed"] = df["dist"] / df["delta_t"]

            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=["delta_t", "dist", "speed"])
            df = df[df["delta_t"] > 0]

            if len(df) == 0:
                continue

            all_delta_t.extend(df["delta_t"].tolist())
            all_dist.extend(df["dist"].tolist())
            all_speed.extend(df["speed"].tolist())

            add_stop_durations(df, stop_durations)

        except Exception as e:
            logger.error(f"Error on {file}: {e}")

    # =========================
    # DECISION GRAPHS
    # =========================

    save_cdf(
        stop_durations,
        "CDF of staypoint durations",
        "Staypoint duration (minutes)",
        "01_staypoint_duration_cdf.png",
        thresholds=[5, 8, 10, 15, 20, 180],
        max_value=240
    )

    save_cdf(
        all_delta_t,
        "CDF of time gaps between GPS points",
        "Delta time (seconds, log scale)",
        "02_delta_time_cdf_log.png",
        thresholds=[300, 600, 900, 1800],
        log_x=True
    )

    save_cdf(
        all_dist,
        "CDF of distances between consecutive GPS points",
        "Distance (meters, log scale)",
        "03_distance_cdf_log.png",
        thresholds=[50, 100, 200, 500],
        log_x=True
    )

    save_cdf(
        all_dist,
        "CDF of distances, zoom on GPS noise",
        "Distance (meters)",
        "04_distance_cdf_zoom_50m.png",
        thresholds=[5, 10, 30, 50],
        max_value=50
    )

    save_cdf(
        all_speed,
        "CDF of calculated speeds",
        "Speed (m/s, log scale)",
        "05_speed_cdf_log.png",
        thresholds=[1, 5, 15, 30],
        log_x=True
    )

    save_hist(
        stop_durations,
        "Histogram of staypoint durations",
        "Staypoint duration (minutes)",
        "06_staypoint_duration_hist.png",
        max_value=60
    )

    save_summary(all_delta_t, all_dist, all_speed, stop_durations)

    logger.info("Finished")
    logger.info(f"Outputs: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()