import argparse
import json
import math
import os
from collections import Counter, defaultdict

import folium
import pandas as pd
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen, MiniMap, MousePosition, MeasureControl
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "NetMob25CleanedData", "NetMob25CleanedData")

REFINED_FILE = os.path.join(BASE_DIR, "all_refined_trajectories.csv")
INDIVIDUALS_FILE = os.path.join(DATA_DIR, "individuals_dataset.csv")
DISPLACEMENTS_FILE = os.path.join(DATA_DIR, "displacements_dataset.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "focused_clustering_output")
SELECTED_USERS_FILE = os.path.join(OUTPUT_DIR, "selected_same_zone_users.csv")
FOCUSED_POINTS_FILE = os.path.join(OUTPUT_DIR, "focused_50_refined_trajectories.csv")
GENERAL_CLUSTERS_FILE = os.path.join(OUTPUT_DIR, "focused_general_clusters.csv")
GENERAL_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "focused_general_summary.csv")
GENERAL_SEGMENTS_FILE = os.path.join(OUTPUT_DIR, "focused_general_weighted_segments.csv")
PROFILE_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "focused_profile_summary.csv")
GENERAL_MAP_FILE = os.path.join(OUTPUT_DIR, "map_focused_general_clusters.html")
PROFILE_MAP_FILE = os.path.join(OUTPUT_DIR, "map_focused_profile_clusters.html")

IDF_REFERENCE_LAT = 48.8566
METERS_PER_DEGREE_LAT = 111_320
METERS_PER_DEGREE_LON = 111_320 * math.cos(math.radians(IDF_REFERENCE_LAT))

POINT_COLUMNS = [
    "LATITUDE",
    "LONGITUDE",
    "DATETIME",
    "USER_ID",
    "SOURCE_FILE",
    "TRAJECTORY_ID",
    "TRAJECTORY_UID",
    "POINT_INDEX",
]

COLORS = [
    "#22d3ee",
    "#f97316",
    "#a78bfa",
    "#34d399",
    "#fb7185",
    "#facc15",
    "#60a5fa",
    "#f472b6",
    "#2dd4bf",
    "#c084fc",
]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def cell_indexes(lat, lon, cell_size_m):
    return (
        math.floor((lat * METERS_PER_DEGREE_LAT) / cell_size_m),
        math.floor((lon * METERS_PER_DEGREE_LON) / cell_size_m),
    )


def cell_id(lat, lon, cell_size_m):
    y, x = cell_indexes(lat, lon, cell_size_m)
    return f"{y}:{x}"


def cell_center(cell, cell_size_m):
    y, x = [int(value) for value in str(cell).split(":")]
    lat = ((y + 0.5) * cell_size_m) / METERS_PER_DEGREE_LAT
    lon = ((x + 0.5) * cell_size_m) / METERS_PER_DEGREE_LON
    return lat, lon


def user_id_from_csv_name(csv_name):
    return os.path.splitext(os.path.basename(str(csv_name)))[0]


def age_group(age):
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if age < 18:
        return "<18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


def nb_car_group(value):
    if pd.isna(value):
        return "Unknown"
    value = int(value)
    if value <= 0:
        return "0 car"
    if value == 1:
        return "1 car"
    return "2+ cars"


def bool_label(value):
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, str):
        return "Yes" if value.strip().lower() in {"true", "1", "yes"} else "No"
    return "Yes" if bool(value) else "No"


def summarize_users_from_refined(input_path, chunksize):
    print("Lecture du grand CSV raffine pour resumer les utilisateurs...")
    rows = []

    usecols = ["LATITUDE", "LONGITUDE", "USER_ID", "TRAJECTORY_UID"]
    for chunk in tqdm(
        pd.read_csv(input_path, usecols=usecols, chunksize=chunksize),
        desc="Resume utilisateurs",
        unit="chunk",
    ):
        chunk = chunk.dropna(subset=["LATITUDE", "LONGITUDE", "USER_ID"])
        grouped = chunk.groupby("USER_ID", sort=False)
        part = grouped.agg(
            N_POINTS=("LATITUDE", "size"),
            SUM_LAT=("LATITUDE", "sum"),
            SUM_LON=("LONGITUDE", "sum"),
            MIN_LAT=("LATITUDE", "min"),
            MAX_LAT=("LATITUDE", "max"),
            MIN_LON=("LONGITUDE", "min"),
            MAX_LON=("LONGITUDE", "max"),
            N_TRAJECTORIES=("TRAJECTORY_UID", "nunique"),
        )
        rows.append(part.reset_index())

    summary = pd.concat(rows, ignore_index=True)
    summary = (
        summary.groupby("USER_ID", as_index=False)
        .agg(
            N_POINTS=("N_POINTS", "sum"),
            SUM_LAT=("SUM_LAT", "sum"),
            SUM_LON=("SUM_LON", "sum"),
            MIN_LAT=("MIN_LAT", "min"),
            MAX_LAT=("MAX_LAT", "max"),
            MIN_LON=("MIN_LON", "min"),
            MAX_LON=("MAX_LON", "max"),
            N_TRAJECTORIES=("N_TRAJECTORIES", "sum"),
        )
    )
    summary["MEAN_LAT"] = summary["SUM_LAT"] / summary["N_POINTS"]
    summary["MEAN_LON"] = summary["SUM_LON"] / summary["N_POINTS"]
    return summary.drop(columns=["SUM_LAT", "SUM_LON"])


def choose_same_zone_users(user_summary, target_users, min_points, zone_cell_sizes):
    candidates = user_summary[user_summary["N_POINTS"] >= min_points].copy()
    if len(candidates) < target_users:
        candidates = user_summary.copy()

    best = None
    for cell_size in zone_cell_sizes:
        candidates["ZONE_CELL"] = candidates.apply(
            lambda row: cell_id(row.MEAN_LAT, row.MEAN_LON, cell_size),
            axis=1,
        )
        grouped = (
            candidates.groupby("ZONE_CELL")
            .agg(
                N_USERS=("USER_ID", "nunique"),
                N_POINTS=("N_POINTS", "sum"),
                N_TRAJECTORIES=("N_TRAJECTORIES", "sum"),
            )
            .reset_index()
            .sort_values(["N_USERS", "N_TRAJECTORIES", "N_POINTS"], ascending=False)
        )
        eligible = grouped[grouped["N_USERS"] >= target_users]
        if not eligible.empty:
            best = eligible.iloc[0].to_dict()
            best["ZONE_CELL_SIZE_M"] = cell_size
            break

    if best is None:
        cell_size = zone_cell_sizes[-1]
        candidates["ZONE_CELL"] = candidates.apply(
            lambda row: cell_id(row.MEAN_LAT, row.MEAN_LON, cell_size),
            axis=1,
        )
        best = (
            candidates.groupby("ZONE_CELL")
            .agg(
                N_USERS=("USER_ID", "nunique"),
                N_POINTS=("N_POINTS", "sum"),
                N_TRAJECTORIES=("N_TRAJECTORIES", "sum"),
            )
            .reset_index()
            .sort_values(["N_USERS", "N_TRAJECTORIES", "N_POINTS"], ascending=False)
            .iloc[0]
            .to_dict()
        )
        best["ZONE_CELL_SIZE_M"] = cell_size

    selected = candidates[candidates["ZONE_CELL"] == best["ZONE_CELL"]].copy()
    selected = selected.sort_values(["N_TRAJECTORIES", "N_POINTS"], ascending=False)
    selected = selected.head(target_users).copy()
    selected["ZONE_CELL_SIZE_M"] = best["ZONE_CELL_SIZE_M"]

    print(
        f"Zone choisie : cellule {best['ZONE_CELL']} "
        f"({int(best['ZONE_CELL_SIZE_M'])} m), "
        f"{len(selected)} utilisateurs selectionnes."
    )
    return selected


def extract_selected_points(input_path, selected_users, output_path, chunksize):
    selected_user_set = set(selected_users["USER_ID"].astype(str))
    wrote_header = False
    total_rows = 0

    print("Extraction des points raffines des 50 utilisateurs...")
    for chunk in tqdm(
        pd.read_csv(input_path, usecols=lambda col: col in POINT_COLUMNS, chunksize=chunksize),
        desc="Extraction points",
        unit="chunk",
    ):
        chunk["USER_ID"] = chunk["USER_ID"].astype(str)
        filtered = chunk[chunk["USER_ID"].isin(selected_user_set)].copy()
        if filtered.empty:
            continue
        filtered.to_csv(
            output_path,
            index=False,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
        )
        wrote_header = True
        total_rows += len(filtered)

    if total_rows == 0:
        raise ValueError("Aucun point extrait pour les utilisateurs selectionnes.")

    return total_rows


def load_focused_points(path):
    points = pd.read_csv(path)
    points = points.dropna(subset=["LATITUDE", "LONGITUDE", "USER_ID", "TRAJECTORY_UID"])
    points["USER_ID"] = points["USER_ID"].astype(str)

    sort_columns = ["TRAJECTORY_UID"]
    if "POINT_INDEX" in points.columns:
        sort_columns.append("POINT_INDEX")
    elif "DATETIME" in points.columns:
        sort_columns.append("DATETIME")
    return points.sort_values(sort_columns).reset_index(drop=True)


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def build_trajectory_features(points_df, cell_size_m):
    rows = []
    grouped = points_df.groupby("TRAJECTORY_UID", sort=False)

    for trajectory_uid, traj_df in tqdm(
        grouped,
        total=points_df["TRAJECTORY_UID"].nunique(),
        desc="Features trajets",
        unit="trajet",
    ):
        cells = [
            cell_id(row.LATITUDE, row.LONGITUDE, cell_size_m)
            for row in traj_df.itertuples(index=False)
        ]
        compressed = []
        for value in cells:
            if not compressed or compressed[-1] != value:
                compressed.append(value)
        unique_cells = sorted(set(compressed))
        if len(unique_cells) < 4:
            continue
        rows.append(
            {
                "TRAJECTORY_UID": trajectory_uid,
                "USER_ID": str(traj_df["USER_ID"].iloc[0]),
                "N_POINTS": len(traj_df),
                "N_CELLS": len(unique_cells),
                "START_CELL": compressed[0],
                "END_CELL": compressed[-1],
                "CELL_SEQUENCE": " ".join(compressed),
                "CELL_SET": " ".join(unique_cells),
            }
        )

    return pd.DataFrame(rows)


def jaccard(left, right):
    shared = len(left & right)
    if shared == 0:
        return 0.0, 0
    return shared / len(left | right), shared


def cluster_trajectories(features, threshold, min_shared_cells, no_od_filter):
    if features.empty:
        return features.copy()

    features = features.copy().reset_index(drop=True)
    features["CELL_SET_OBJ"] = features["CELL_SET"].apply(lambda value: set(str(value).split()))

    ids = features["TRAJECTORY_UID"].tolist()
    union_find = UnionFind(ids)
    buckets = defaultdict(list)
    cell_sets = dict(zip(features["TRAJECTORY_UID"], features["CELL_SET_OBJ"]))

    for row in features.itertuples(index=False):
        key = "ALL" if no_od_filter else (row.START_CELL, row.END_CELL)
        buckets[key].append(row.TRAJECTORY_UID)

    for bucket_ids in tqdm(
        buckets.values(),
        total=len(buckets),
        desc="Comparaison trajets",
        unit="bucket",
    ):
        for i, left_uid in enumerate(bucket_ids):
            left_cells = cell_sets[left_uid]
            for right_uid in bucket_ids[i + 1:]:
                score, shared = jaccard(left_cells, cell_sets[right_uid])
                if shared >= min_shared_cells and score >= threshold:
                    union_find.union(left_uid, right_uid)

    root_to_cluster = {}
    cluster_ids = []
    for trajectory_uid in ids:
        root = union_find.find(trajectory_uid)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        cluster_ids.append(root_to_cluster[root])

    features["CLUSTER_ID"] = cluster_ids
    return features.drop(columns=["CELL_SET_OBJ"])


def summarize_clusters(clusters):
    if clusters.empty:
        return pd.DataFrame()
    return (
        clusters.groupby("CLUSTER_ID")
        .agg(
            N_TRAJECTORIES=("TRAJECTORY_UID", "nunique"),
            N_USERS=("USER_ID", "nunique"),
            AVG_POINTS=("N_POINTS", "mean"),
            AVG_CELLS=("N_CELLS", "mean"),
            START_CELL=("START_CELL", "first"),
            END_CELL=("END_CELL", "first"),
        )
        .reset_index()
        .sort_values(["N_USERS", "N_TRAJECTORIES", "AVG_CELLS"], ascending=False)
    )


def build_weighted_segments(points_df, cluster_map=None, cell_size_m=100):
    rows = []
    cluster_map = cluster_map or {}

    for trajectory_uid, traj_df in tqdm(
        points_df.groupby("TRAJECTORY_UID", sort=False),
        total=points_df["TRAJECTORY_UID"].nunique(),
        desc="Segments ponderes",
        unit="trajet",
    ):
        user_id = str(traj_df["USER_ID"].iloc[0])
        cluster_id_value = cluster_map.get(trajectory_uid, -1)
        seen = {}
        previous_cell = None
        previous_lat = None
        previous_lon = None

        for row in traj_df.itertuples(index=False):
            current_cell = cell_id(row.LATITUDE, row.LONGITUDE, cell_size_m)
            if previous_cell is None:
                previous_cell = current_cell
                previous_lat = row.LATITUDE
                previous_lon = row.LONGITUDE
                continue
            if current_cell == previous_cell:
                previous_lat = row.LATITUDE
                previous_lon = row.LONGITUDE
                continue

            key = (previous_cell, current_cell, cluster_id_value)
            if key not in seen:
                seen[key] = {
                    "FROM_CELL": previous_cell,
                    "TO_CELL": current_cell,
                    "CLUSTER_ID": cluster_id_value,
                    "FROM_LATITUDE": previous_lat,
                    "FROM_LONGITUDE": previous_lon,
                    "TO_LATITUDE": row.LATITUDE,
                    "TO_LONGITUDE": row.LONGITUDE,
                }

            previous_cell = current_cell
            previous_lat = row.LATITUDE
            previous_lon = row.LONGITUDE

        for transition in seen.values():
            transition["USER_ID"] = user_id
            transition["TRAJECTORY_UID"] = trajectory_uid
            rows.append(transition)

    if not rows:
        return pd.DataFrame()

    raw_segments = pd.DataFrame(rows)
    grouped = [
        "FROM_CELL",
        "TO_CELL",
        "CLUSTER_ID",
    ]
    return (
        raw_segments.groupby(grouped)
        .agg(
            FROM_LATITUDE=("FROM_LATITUDE", "mean"),
            FROM_LONGITUDE=("FROM_LONGITUDE", "mean"),
            TO_LATITUDE=("TO_LATITUDE", "mean"),
            TO_LONGITUDE=("TO_LONGITUDE", "mean"),
            N_USERS=("USER_ID", "nunique"),
            N_TRAJECTORIES=("TRAJECTORY_UID", "nunique"),
        )
        .reset_index()
        .sort_values(["N_USERS", "N_TRAJECTORIES"], ascending=False)
    )


def color_for_weight(value, max_value):
    if max_value <= 1:
        return "#22d3ee"
    ratio = value / max_value
    if ratio >= 0.75:
        return "#ef4444"
    if ratio >= 0.50:
        return "#f97316"
    if ratio >= 0.30:
        return "#facc15"
    return "#22d3ee"


def add_weighted_segment_layer(map_object, segments, name, show, color=None, max_segments=2500):
    if segments.empty:
        return 0

    layer = folium.FeatureGroup(name=name, show=show)
    map_segments = (
        segments.sort_values(["N_USERS", "N_TRAJECTORIES"], ascending=False)
        .head(max_segments)
        .copy()
    )
    max_users = max(1, int(map_segments["N_USERS"].max()))

    for row in map_segments.itertuples(index=False):
        segment_color = color or color_for_weight(row.N_USERS, max_users)
        ratio = row.N_USERS / max_users
        weight = 2.0 + 8.0 * ratio
        opacity = 0.35 + 0.55 * ratio
        points = [
            [row.FROM_LATITUDE, row.FROM_LONGITUDE],
            [row.TO_LATITUDE, row.TO_LONGITUDE],
        ]
        tooltip = (
            f"{row.N_USERS} utilisateurs | "
            f"{row.N_TRAJECTORIES} trajets"
        )
        folium.PolyLine(
            points,
            color="#020617",
            weight=weight + 2,
            opacity=0.18,
            line_cap="butt",
        ).add_to(layer)
        folium.PolyLine(
            points,
            color=segment_color,
            weight=weight,
            opacity=opacity,
            tooltip=tooltip,
            line_cap="butt",
        ).add_to(layer)

    layer.add_to(map_object)
    return len(map_segments)


def add_continuous_trajectory_layer(
    map_object,
    points_df,
    name,
    show,
    color,
    max_trajectories=100,
    point_step=3,
    weight=3.2,
    opacity=0.68,
):
    if points_df.empty:
        return 0

    layer = folium.FeatureGroup(name=name, show=show)
    grouped = []

    for trajectory_uid, trajectory_df in points_df.groupby("TRAJECTORY_UID", sort=False):
        if len(trajectory_df) < 2:
            continue
        grouped.append((trajectory_uid, len(trajectory_df), trajectory_df))

    grouped.sort(key=lambda item: item[1], reverse=True)
    selected = grouped[:max_trajectories]

    for trajectory_uid, _, trajectory_df in selected:
        sampled = trajectory_df.iloc[::max(1, point_step)]
        coords = sampled[["LATITUDE", "LONGITUDE"]].values.tolist()
        if len(coords) < 2:
            coords = trajectory_df[["LATITUDE", "LONGITUDE"]].values.tolist()
        if len(coords) < 2:
            continue

        user_id = trajectory_df["USER_ID"].iloc[0]
        tooltip = f"{trajectory_uid} | utilisateur {user_id}"

        folium.PolyLine(
            coords,
            color="#020617",
            weight=weight + 2.4,
            opacity=0.38,
            smooth_factor=1.4,
            tooltip=tooltip,
        ).add_to(layer)
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=opacity,
            smooth_factor=1.4,
            tooltip=tooltip,
        ).add_to(layer)

    layer.add_to(map_object)
    return len(selected)


def choose_cluster_medoid(cluster_rows):
    if cluster_rows.empty:
        return None

    rows = cluster_rows.copy()
    rows["CELL_SET_OBJ"] = rows["CELL_SET"].apply(lambda value: set(str(value).split()))
    records = rows.to_dict("records")

    best_record = None
    best_score = -1

    for candidate in records:
        candidate_cells = candidate["CELL_SET_OBJ"]
        if not candidate_cells:
            continue

        total_similarity = 0
        for other in records:
            other_cells = other["CELL_SET_OBJ"]
            if not other_cells:
                continue
            similarity, _ = jaccard(candidate_cells, other_cells)
            total_similarity += similarity

        # Tie-breaker: prefer the richer trajectory, not a tiny shortcut.
        score = total_similarity + (candidate["N_CELLS"] * 0.001)
        if score > best_score:
            best_score = score
            best_record = candidate

    return best_record


def add_cluster_representative_layer(
    map_object,
    points_df,
    clusters_df,
    summary_df,
    top_cluster_ids,
    args,
):
    if not top_cluster_ids:
        return 0

    layer = folium.FeatureGroup(name="Trajets representatifs des clusters", show=True)
    summary_by_cluster = summary_df.set_index("CLUSTER_ID").to_dict("index")
    max_users = max(
        1,
        int(summary_df[summary_df["CLUSTER_ID"].isin(top_cluster_ids)]["N_USERS"].max()),
    )
    displayed = 0

    for rank, cluster_id_value in enumerate(top_cluster_ids):
        cluster_rows = clusters_df[clusters_df["CLUSTER_ID"] == cluster_id_value]
        medoid = choose_cluster_medoid(cluster_rows)
        if medoid is None:
            continue

        trajectory_uid = medoid["TRAJECTORY_UID"]
        trajectory_df = points_df[points_df["TRAJECTORY_UID"] == trajectory_uid]
        if len(trajectory_df) < 2:
            continue

        sampled = trajectory_df.iloc[::max(1, args.map_trajectory_point_step)]
        coords = sampled[["LATITUDE", "LONGITUDE"]].values.tolist()
        if len(coords) < 2:
            coords = trajectory_df[["LATITUDE", "LONGITUDE"]].values.tolist()
        if len(coords) < 2:
            continue

        summary = summary_by_cluster.get(cluster_id_value, {})
        n_users = int(summary.get("N_USERS", 0))
        n_trajectories = int(summary.get("N_TRAJECTORIES", 0))
        user_ratio = n_users / max_users
        weight = args.map_trajectory_weight + 5.0 * user_ratio
        color = COLORS[rank % len(COLORS)]
        tooltip = (
            f"Cluster {cluster_id_value} | "
            f"{n_users} utilisateurs | "
            f"{n_trajectories} trajets similaires | "
            f"representant: {trajectory_uid}"
        )

        folium.PolyLine(
            coords,
            color="#020617",
            weight=weight + 3.0,
            opacity=0.50,
            smooth_factor=1.6,
            tooltip=tooltip,
        ).add_to(layer)
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=args.map_trajectory_opacity,
            smooth_factor=1.6,
            tooltip=tooltip,
        ).add_to(layer)
        displayed += 1

    layer.add_to(map_object)
    return displayed


def add_profile_representative_layer(
    map_object,
    profile_points,
    name,
    show,
    color,
    args,
):
    if profile_points.empty:
        return 0

    features = build_trajectory_features(profile_points, args.cluster_cell_size_m)
    if features.empty:
        return 0

    clusters = cluster_trajectories(
        features,
        threshold=args.jaccard_threshold,
        min_shared_cells=args.min_shared_cells,
        no_od_filter=not args.od_filter,
    )
    summary = summarize_clusters(clusters)
    if summary.empty:
        return 0

    eligible = summary[
        (summary["N_USERS"] >= args.profile_min_cluster_users)
        & (summary["N_TRAJECTORIES"] >= args.profile_min_cluster_trajectories)
    ].head(args.profile_map_top_clusters)

    if eligible.empty:
        return 0

    layer = folium.FeatureGroup(name=name, show=show)
    max_users = max(1, int(eligible["N_USERS"].max()))
    displayed = 0

    for _, summary_row in eligible.iterrows():
        cluster_id_value = summary_row["CLUSTER_ID"]
        cluster_rows = clusters[clusters["CLUSTER_ID"] == cluster_id_value]
        medoid = choose_cluster_medoid(cluster_rows)
        if medoid is None:
            continue

        trajectory_uid = medoid["TRAJECTORY_UID"]
        trajectory_df = profile_points[profile_points["TRAJECTORY_UID"] == trajectory_uid]
        if len(trajectory_df) < 2:
            continue

        sampled = trajectory_df.iloc[::max(1, args.map_trajectory_point_step)]
        coords = sampled[["LATITUDE", "LONGITUDE"]].values.tolist()
        if len(coords) < 2:
            coords = trajectory_df[["LATITUDE", "LONGITUDE"]].values.tolist()
        if len(coords) < 2:
            continue

        n_users = int(summary_row["N_USERS"])
        n_trajectories = int(summary_row["N_TRAJECTORIES"])
        user_ratio = n_users / max_users
        weight = args.map_trajectory_weight + 4.0 * user_ratio
        tooltip = (
            f"{name} | cluster {cluster_id_value} | "
            f"{n_users} utilisateurs | {n_trajectories} trajets similaires | "
            f"representant: {trajectory_uid}"
        )

        folium.PolyLine(
            coords,
            color="#020617",
            weight=weight + 3.0,
            opacity=0.48,
            smooth_factor=1.6,
            tooltip=tooltip,
        ).add_to(layer)
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=args.map_trajectory_opacity,
            smooth_factor=1.6,
            tooltip=tooltip,
        ).add_to(layer)
        displayed += 1

    if displayed:
        layer.add_to(map_object)

    return displayed


def collect_profile_representative_records(profile_points, criterion, value, color, args):
    if profile_points.empty:
        return []

    features = build_trajectory_features(profile_points, args.cluster_cell_size_m)
    if features.empty:
        return []

    clusters = cluster_trajectories(
        features,
        threshold=args.jaccard_threshold,
        min_shared_cells=args.min_shared_cells,
        no_od_filter=not args.od_filter,
    )
    summary = summarize_clusters(clusters)
    if summary.empty:
        return []

    eligible = summary[
        (summary["N_USERS"] >= args.profile_min_cluster_users)
        & (summary["N_TRAJECTORIES"] >= args.profile_min_cluster_trajectories)
    ].head(args.profile_map_top_clusters)

    records = []
    for _, summary_row in eligible.iterrows():
        cluster_id_value = summary_row["CLUSTER_ID"]
        cluster_rows = clusters[clusters["CLUSTER_ID"] == cluster_id_value]
        medoid = choose_cluster_medoid(cluster_rows)
        if medoid is None:
            continue

        trajectory_uid = medoid["TRAJECTORY_UID"]
        trajectory_df = profile_points[profile_points["TRAJECTORY_UID"] == trajectory_uid]
        if len(trajectory_df) < 2:
            continue

        sampled = trajectory_df.iloc[::max(1, args.map_trajectory_point_step)]
        coords = sampled[["LATITUDE", "LONGITUDE"]].values.tolist()
        if len(coords) < 2:
            coords = trajectory_df[["LATITUDE", "LONGITUDE"]].values.tolist()
        if len(coords) < 2:
            continue

        records.append(
            {
                "criterion": str(criterion),
                "value": str(value),
                "cluster_id": int(cluster_id_value),
                "n_users": int(summary_row["N_USERS"]),
                "n_trajectories": int(summary_row["N_TRAJECTORIES"]),
                "representative": str(trajectory_uid),
                "color": color,
                "coords": coords,
            }
        )

    return records


def add_profile_filter_ui(map_object, records, map_name):
    records_json = json.dumps(records, ensure_ascii=True)
    ui_html = f"""
{{% macro html(this, kwargs) %}}
<style>
#profile-filter-panel {{
  position: fixed;
  top: 18px;
  left: 18px;
  z-index: 9999;
  width: 360px;
  background: rgba(15, 23, 42, 0.94);
  color: #e5eefb;
  font-family: Arial, sans-serif;
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 8px;
  box-shadow: 0 18px 40px rgba(0,0,0,.32);
  overflow: hidden;
}}
#profile-filter-panel .head {{
  padding: 14px 16px;
  font-size: 16px;
  font-weight: 700;
  background: linear-gradient(90deg, rgba(34,211,238,.22), rgba(249,115,22,.20));
}}
#profile-filter-panel .body {{ padding: 12px 16px 14px; }}
#profile-filter-panel label {{
  display: block;
  margin-top: 10px;
  margin-bottom: 5px;
  color: #aab8cc;
  font-size: 12px;
}}
#profile-filter-panel select {{
  width: 100%;
  background: #0f172a;
  color: #f8fafc;
  border: 1px solid rgba(148, 163, 184, 0.45);
  border-radius: 6px;
  padding: 8px;
  font-size: 13px;
}}
#profile-filter-panel .metrics {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 12px;
}}
#profile-filter-panel .metric {{
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 6px;
  padding: 8px;
}}
#profile-filter-panel .metric span {{
  display: block;
  color: #94a3b8;
  font-size: 10px;
  text-transform: uppercase;
}}
#profile-filter-panel .metric b {{
  display: block;
  margin-top: 3px;
  color: #f8fafc;
  font-family: Consolas, monospace;
  font-size: 15px;
}}
</style>
<div id="profile-filter-panel">
  <div class="head">Clusters par profil</div>
  <div class="body">
    <label for="profile-criterion">Critere</label>
    <select id="profile-criterion"></select>
    <label for="profile-value">Valeur</label>
    <select id="profile-value"></select>
    <div class="metrics">
      <div class="metric"><span>Lignes</span><b id="profile-line-count">0</b></div>
      <div class="metric"><span>Trajets</span><b id="profile-trip-count">0</b></div>
      <div class="metric"><span>Utilisateurs max</span><b id="profile-user-count">0</b></div>
      <div class="metric"><span>Lecture</span><b>1 ligne / cluster</b></div>
    </div>
  </div>
</div>
{{% endmacro %}}
"""
    panel = MacroElement()
    panel._template = Template(ui_html)
    map_object.get_root().add_child(panel)

    js = f"""
<script>
(function() {{
  const PROFILE_RECORDS = {records_json};
  const mapName = "{map_name}";
  let profileLayer = null;

  function waitForProfileMap() {{
    if (typeof window[mapName] !== "undefined" &&
        document.getElementById("profile-criterion") &&
        document.getElementById("profile-value")) {{
      bootProfileFilter(window[mapName]);
    }} else {{
      setTimeout(waitForProfileMap, 100);
    }}
  }}

  function unique(values) {{
    return Array.from(new Set(values)).sort();
  }}

  function fillSelect(select, values) {{
    select.innerHTML = "";
    values.forEach(function(value) {{
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    }});
  }}

  function render(map) {{
    const criterion = document.getElementById("profile-criterion").value;
    let value = document.getElementById("profile-value").value;
    let selected = PROFILE_RECORDS.filter(function(record) {{
      return record.criterion === criterion && record.value === value;
    }});

    if (!selected.length) {{
      const fallbackValues = unique(PROFILE_RECORDS
        .filter(function(record) {{ return record.criterion === criterion; }})
        .map(function(record) {{ return record.value; }}));
      if (fallbackValues.length) {{
        value = fallbackValues[0];
        document.getElementById("profile-value").value = value;
        selected = PROFILE_RECORDS.filter(function(record) {{
          return record.criterion === criterion && record.value === value;
        }});
      }}
    }}

    if (profileLayer) map.removeLayer(profileLayer);
    profileLayer = L.layerGroup();

    let totalTrips = 0;
    let maxUsers = 0;
    const bounds = [];

    selected.forEach(function(record) {{
      totalTrips += record.n_trajectories;
      maxUsers = Math.max(maxUsers, record.n_users);
      const tooltip = (
        record.criterion + " = " + record.value +
        " | cluster " + record.cluster_id +
        " | " + record.n_users + " utilisateurs" +
        " | " + record.n_trajectories + " trajets similaires" +
        " | representant: " + record.representative
      );
      const ratio = Math.max(0.25, record.n_users / Math.max(1, maxUsers));
      const weight = 4 + 5 * ratio;
      L.polyline(record.coords, {{
        color: "#020617",
        weight: weight + 3,
        opacity: 0.48,
        smoothFactor: 1.6
      }}).bindTooltip(tooltip).addTo(profileLayer);
      L.polyline(record.coords, {{
        color: record.color,
        weight: weight,
        opacity: 0.72,
        smoothFactor: 1.6
      }}).bindTooltip(tooltip).addTo(profileLayer);
      record.coords.forEach(function(coord) {{ bounds.push(coord); }});
    }});

    profileLayer.addTo(map);
    document.getElementById("profile-line-count").textContent = selected.length;
    document.getElementById("profile-trip-count").textContent = totalTrips;
    document.getElementById("profile-user-count").textContent = maxUsers;

    if (bounds.length) {{
      map.fitBounds(bounds, {{padding: [35, 35]}});
    }}
  }}

  function updateValues(map) {{
    const criterion = document.getElementById("profile-criterion").value;
    const values = unique(PROFILE_RECORDS
      .filter(function(record) {{ return record.criterion === criterion; }})
      .map(function(record) {{ return record.value; }}));
    fillSelect(document.getElementById("profile-value"), values);
    render(map);
  }}

  function bootProfileFilter(map) {{
    if (!PROFILE_RECORDS.length) return;
    const criteria = unique(PROFILE_RECORDS.map(function(record) {{ return record.criterion; }}));
    fillSelect(document.getElementById("profile-criterion"), criteria);
    document.getElementById("profile-criterion").addEventListener("change", function() {{
      updateValues(map);
    }});
    document.getElementById("profile-value").addEventListener("change", function() {{
      render(map);
    }});
    updateValues(map);
  }}

  waitForProfileMap();
}})();
</script>
"""
    map_object.get_root().html.add_child(folium.Element(js))


def add_panel(map_object, title, rows):
    row_html = "\n".join(
        f"<div class=\"metric\"><span>{label}</span><b>{value}</b></div>"
        for label, value in rows
    )
    panel_html = f"""
{{% macro html(this, kwargs) %}}
<style>
#focus-panel {{
  position: fixed;
  top: 18px;
  left: 18px;
  z-index: 9999;
  width: 350px;
  background: rgba(15, 23, 42, 0.92);
  color: #e5eefb;
  font-family: Arial, sans-serif;
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 8px;
  box-shadow: 0 18px 40px rgba(0,0,0,.32);
  overflow: hidden;
}}
#focus-panel .head {{
  padding: 14px 16px;
  font-size: 16px;
  font-weight: 700;
  background: linear-gradient(90deg, rgba(34,211,238,.22), rgba(249,115,22,.20));
}}
#focus-panel .body {{
  padding: 10px 16px 14px;
}}
#focus-panel .metric {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 8px 0;
  border-top: 1px solid rgba(148, 163, 184, 0.22);
  font-size: 12px;
}}
#focus-panel .metric:first-child {{ border-top: 0; }}
#focus-panel .metric span {{ color: #aab8cc; }}
#focus-panel .metric b {{ color: #f8fafc; font-family: Consolas, monospace; }}
</style>
<div id="focus-panel">
  <div class="head">{title}</div>
  <div class="body">{row_html}</div>
</div>
{{% endmacro %}}
"""
    panel = MacroElement()
    panel._template = Template(panel_html)
    map_object.get_root().add_child(panel)


def build_general_map(points, clusters, summary, segments, selected_users, args):
    center = [points["LATITUDE"].mean(), points["LONGITUDE"].mean()]
    map_object = folium.Map(
        location=center,
        zoom_start=12,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer("CartoDB dark_matter", name="Fond sombre", control=True).add_to(map_object)
    folium.TileLayer("CartoDB positron", name="Fond clair", control=True).add_to(map_object)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(map_object)

    top_clusters = (
        summary[
            (summary["N_USERS"] >= args.map_min_cluster_users)
            & (summary["N_TRAJECTORIES"] >= args.map_min_cluster_trajectories)
        ]
        .head(args.map_top_clusters)
        ["CLUSTER_ID"]
        .tolist()
    )

    displayed_lines = add_cluster_representative_layer(
        map_object,
        points,
        clusters,
        summary,
        top_clusters,
        args,
    )

    for rank, cluster_id_value in enumerate(top_clusters):
        cluster_uids = clusters[
            clusters["CLUSTER_ID"] == cluster_id_value
        ].sort_values(["N_POINTS", "N_CELLS"], ascending=False)["TRAJECTORY_UID"]
        color = COLORS[rank % len(COLORS)]

        cluster_segments = segments[
            (segments["CLUSTER_ID"] == cluster_id_value)
            & (segments["N_USERS"] >= args.min_segment_users)
        ]
        add_weighted_segment_layer(
            map_object,
            cluster_segments,
            f"Segments ponderes - cluster {cluster_id_value}",
            False,
            color=color,
            max_segments=args.map_max_segments_per_cluster,
        )

        if args.include_variant_layers:
            cluster_points = points[points["TRAJECTORY_UID"].isin(cluster_uids)].copy()
            add_continuous_trajectory_layer(
                map_object,
                cluster_points,
                f"Variantes individuelles - cluster {cluster_id_value}",
                False,
                color=color,
                max_trajectories=args.map_max_trajectories_per_cluster,
                point_step=args.map_trajectory_point_step,
                weight=args.map_trajectory_weight,
                opacity=0.42,
            )

    all_common = segments[
        (segments["N_USERS"] >= args.min_segment_users)
        & (segments["N_TRAJECTORIES"] >= args.min_segment_trajectories)
    ].copy()
    displayed_segments = add_weighted_segment_layer(
        map_object,
        all_common,
        "Segments ponderes - tous profils",
        False,
        max_segments=args.map_max_segments,
    )

    Fullscreen().add_to(map_object)
    MiniMap(toggle_display=True, position="bottomright").add_to(map_object)
    MeasureControl().add_to(map_object)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordonnees",
        num_digits=5,
    ).add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)

    map_object.fit_bounds(
        [
            [points["LATITUDE"].min(), points["LONGITUDE"].min()],
            [points["LATITUDE"].max(), points["LONGITUDE"].max()],
        ]
    )

    add_panel(
        map_object,
        "Clustering general - zone focalisee",
        [
            ("Utilisateurs", f"{selected_users['USER_ID'].nunique()}"),
            ("Trajets raffines", f"{points['TRAJECTORY_UID'].nunique():,}"),
            ("Points GPS", f"{len(points):,}"),
            ("Clusters affichables", f"{len(top_clusters)}"),
            ("Lignes representatives", f"{displayed_lines:,}"),
            ("Segments caches", f"{displayed_segments:,}"),
            ("Cellule selection", f"{int(selected_users['ZONE_CELL_SIZE_M'].iloc[0])} m"),
        ],
    )

    map_object.save(args.general_map_output)
    print(f"Carte generale       : {os.path.abspath(args.general_map_output)}")


def load_profiles(selected_users):
    individuals = pd.read_csv(INDIVIDUALS_FILE)
    individuals["ID"] = individuals["ID"].astype(str)
    profiles = selected_users[["USER_ID"]].merge(
        individuals,
        left_on="USER_ID",
        right_on="ID",
        how="left",
    )
    profiles["AGE_GROUP"] = profiles["AGE"].apply(age_group) if "AGE" in profiles else "Unknown"
    profiles["NB_CAR_GROUP"] = profiles["NB_CAR"].apply(nb_car_group) if "NB_CAR" in profiles else "Unknown"

    for column in ["NAVIGO_SUB", "DRIVING_LICENCE", "BIKE", "TWO_WHEELER"]:
        if column in profiles:
            profiles[column] = profiles[column].apply(bool_label)

    if os.path.exists(DISPLACEMENTS_FILE):
        displacements = pd.read_csv(DISPLACEMENTS_FILE, usecols=["ID", "Main_Mode"])
        displacements = displacements.dropna(subset=["ID", "Main_Mode"])
        displacements["ID"] = displacements["ID"].astype(str)
        dominant = (
            displacements.groupby("ID")["Main_Mode"]
            .agg(lambda values: values.value_counts().index[0])
            .reset_index()
            .rename(columns={"Main_Mode": "DOMINANT_MODE"})
        )
        profiles = profiles.merge(dominant, left_on="USER_ID", right_on="ID", how="left", suffixes=("", "_MODE"))
    else:
        profiles["DOMINANT_MODE"] = "Unknown"

    profiles["DOMINANT_MODE"] = profiles["DOMINANT_MODE"].fillna("Unknown")
    return profiles


def build_profile_summary(points, profiles, args):
    profile_rows = []
    criteria = [
        "SEX",
        "AGE_GROUP",
        "NAVIGO_SUB",
        "DRIVING_LICENCE",
        "NB_CAR_GROUP",
        "DOMINANT_MODE",
    ]

    for criterion in criteria:
        if criterion not in profiles.columns:
            continue
        for value, group in profiles.groupby(criterion, dropna=False):
            user_ids = set(group["USER_ID"].astype(str))
            if len(user_ids) < args.profile_min_users:
                continue
            profile_points = points[points["USER_ID"].astype(str).isin(user_ids)]
            if profile_points["TRAJECTORY_UID"].nunique() < args.profile_min_trajectories:
                continue
            profile_rows.append(
                {
                    "CRITERION": criterion,
                    "VALUE": value,
                    "N_USERS": len(user_ids),
                    "N_TRAJECTORIES": profile_points["TRAJECTORY_UID"].nunique(),
                    "N_POINTS": len(profile_points),
                }
            )

    return pd.DataFrame(profile_rows).sort_values(
        ["CRITERION", "N_USERS", "N_TRAJECTORIES"],
        ascending=[True, False, False],
    )


def build_profile_map(points, profiles, profile_summary, args):
    center = [points["LATITUDE"].mean(), points["LONGITUDE"].mean()]
    map_object = folium.Map(
        location=center,
        zoom_start=12,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer("CartoDB dark_matter", name="Fond sombre", control=True).add_to(map_object)
    folium.TileLayer("CartoDB positron", name="Fond clair", control=True).add_to(map_object)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(map_object)

    profile_records = []
    if not profile_summary.empty:
        for _, row in profile_summary.head(args.profile_max_layers).iterrows():
            criterion = row["CRITERION"]
            value = row["VALUE"]
            user_ids = set(
                profiles[profiles[criterion].astype(str) == str(value)]["USER_ID"].astype(str)
            )
            profile_points = points[points["USER_ID"].astype(str).isin(user_ids)]
            color = COLORS[len(profile_records) % len(COLORS)]
            profile_records.extend(
                collect_profile_representative_records(
                profile_points,
                    criterion,
                    value,
                    color,
                    args,
                )
            )

    Fullscreen().add_to(map_object)
    MiniMap(toggle_display=True, position="bottomright").add_to(map_object)
    MeasureControl().add_to(map_object)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordonnees",
        num_digits=5,
    ).add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)

    map_object.fit_bounds(
        [
            [points["LATITUDE"].min(), points["LONGITUDE"].min()],
            [points["LATITUDE"].max(), points["LONGITUDE"].max()],
        ]
    )

    add_profile_filter_ui(map_object, profile_records, map_object.get_name())

    map_object.save(args.profile_map_output)
    print(f"Carte profils        : {os.path.abspath(args.profile_map_output)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Selectionne 50 utilisateurs d'une meme zone et cree des cartes de corridors."
    )
    parser.add_argument("--input", default=REFINED_FILE)
    parser.add_argument("--target-users", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=600_000)
    parser.add_argument("--min-points-per-user", type=int, default=500)
    parser.add_argument("--zone-cell-sizes", type=int, nargs="+", default=[4000, 6000, 8000, 10000, 12000, 16000])
    parser.add_argument("--cluster-cell-size-m", type=int, default=350)
    parser.add_argument("--segment-cell-size-m", type=int, default=90)
    parser.add_argument("--jaccard-threshold", type=float, default=0.42)
    parser.add_argument("--min-shared-cells", type=int, default=5)
    parser.add_argument(
        "--od-filter",
        action="store_true",
        help="Compare seulement les trajets avec la meme cellule de depart et d'arrivee.",
    )
    parser.add_argument("--min-segment-users", type=int, default=3)
    parser.add_argument("--min-segment-trajectories", type=int, default=2)
    parser.add_argument("--map-min-cluster-users", type=int, default=2)
    parser.add_argument("--map-min-cluster-trajectories", type=int, default=3)
    parser.add_argument("--map-top-clusters", type=int, default=10)
    parser.add_argument("--map-default-visible-clusters", type=int, default=5)
    parser.add_argument("--map-max-trajectories-per-cluster", type=int, default=35)
    parser.add_argument("--map-trajectory-point-step", type=int, default=4)
    parser.add_argument("--map-trajectory-weight", type=float, default=4.0)
    parser.add_argument("--map-trajectory-opacity", type=float, default=0.62)
    parser.add_argument("--map-max-segments", type=int, default=2600)
    parser.add_argument("--map-max-segments-per-cluster", type=int, default=700)
    parser.add_argument(
        "--include-variant-layers",
        action="store_true",
        help="Ajoute des couches cachees avec tous les trajets individuels de chaque cluster.",
    )
    parser.add_argument("--profile-min-users", type=int, default=7)
    parser.add_argument("--profile-min-trajectories", type=int, default=10)
    parser.add_argument("--profile-min-cluster-users", type=int, default=2)
    parser.add_argument("--profile-min-cluster-trajectories", type=int, default=4)
    parser.add_argument("--profile-min-segment-users", type=int, default=2)
    parser.add_argument("--profile-max-layers", type=int, default=30)
    parser.add_argument("--profile-map-top-clusters", type=int, default=2)
    parser.add_argument("--profile-map-max-trajectories", type=int, default=80)
    parser.add_argument("--profile-map-max-segments", type=int, default=900)
    parser.add_argument("--selected-users-output", default=SELECTED_USERS_FILE)
    parser.add_argument("--focused-output", default=FOCUSED_POINTS_FILE)
    parser.add_argument("--general-output", default=GENERAL_CLUSTERS_FILE)
    parser.add_argument("--general-summary-output", default=GENERAL_SUMMARY_FILE)
    parser.add_argument("--segments-output", default=GENERAL_SEGMENTS_FILE)
    parser.add_argument("--profile-summary-output", default=PROFILE_SUMMARY_FILE)
    parser.add_argument("--general-map-output", default=GENERAL_MAP_FILE)
    parser.add_argument("--profile-map-output", default=PROFILE_MAP_FILE)
    parser.add_argument("--reuse-selection", action="store_true")
    parser.add_argument("--reuse-focused-points", action="store_true")
    parser.add_argument("--no-profile-map", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dir()

    print("\n===== FOCUSED MOBILITY CLUSTERING =====\n")
    print(f"Entree raffinee      : {os.path.abspath(args.input)}")
    print(f"Utilisateurs cible   : {args.target_users}")
    print(f"Sorties              : {os.path.abspath(OUTPUT_DIR)}\n")

    if args.reuse_selection and os.path.exists(args.selected_users_output):
        selected_users = pd.read_csv(args.selected_users_output)
        selected_users["USER_ID"] = selected_users["USER_ID"].astype(str)
    else:
        user_summary = summarize_users_from_refined(args.input, args.chunksize)
        selected_users = choose_same_zone_users(
            user_summary,
            target_users=args.target_users,
            min_points=args.min_points_per_user,
            zone_cell_sizes=args.zone_cell_sizes,
        )
        selected_users.to_csv(args.selected_users_output, index=False)
        print(f"Utilisateurs zone    : {os.path.abspath(args.selected_users_output)}")

    if args.reuse_focused_points and os.path.exists(args.focused_output):
        print(f"Points focalises     : {os.path.abspath(args.focused_output)}")
    else:
        total_rows = extract_selected_points(
            args.input,
            selected_users,
            args.focused_output,
            args.chunksize,
        )
        print(f"Points extraits      : {total_rows:,}")
        print(f"CSV focalise         : {os.path.abspath(args.focused_output)}")

    points = load_focused_points(args.focused_output)
    print(f"Trajets focalises    : {points['TRAJECTORY_UID'].nunique():,}")
    print(f"Utilisateurs         : {points['USER_ID'].nunique():,}\n")

    features = build_trajectory_features(points, args.cluster_cell_size_m)
    clusters = cluster_trajectories(
        features,
        threshold=args.jaccard_threshold,
        min_shared_cells=args.min_shared_cells,
        no_od_filter=not args.od_filter,
    )
    summary = summarize_clusters(clusters)
    clusters.to_csv(args.general_output, index=False)
    summary.to_csv(args.general_summary_output, index=False)

    print("===== CLUSTERS GENERAUX =====")
    print(f"Trajets clusterises  : {len(clusters):,}")
    print(f"Clusters             : {clusters['CLUSTER_ID'].nunique():,}")
    print(f"Resume              : {os.path.abspath(args.general_summary_output)}\n")

    cluster_map = dict(zip(clusters["TRAJECTORY_UID"], clusters["CLUSTER_ID"]))
    segments = build_weighted_segments(
        points,
        cluster_map=cluster_map,
        cell_size_m=args.segment_cell_size_m,
    )
    segments.to_csv(args.segments_output, index=False)
    print(f"Segments ponderes    : {len(segments):,}")
    print(f"CSV segments         : {os.path.abspath(args.segments_output)}")

    build_general_map(points, clusters, summary, segments, selected_users, args)

    profiles = load_profiles(selected_users)
    profile_summary = build_profile_summary(points, profiles, args)
    profile_summary.to_csv(args.profile_summary_output, index=False)
    print(f"Resume profils       : {os.path.abspath(args.profile_summary_output)}")

    if not args.no_profile_map:
        build_profile_map(points, profiles, profile_summary, args)

    print("\nTermine.")


if __name__ == "__main__":
    main()
