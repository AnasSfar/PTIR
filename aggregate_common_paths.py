import os
import glob
import math
import json
import pandas as pd
import folium
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from branca.element import Element
from folium.plugins import Fullscreen, MiniMap, MeasureControl

# =========================
# PARAMÃˆTRES
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "all_refined_trajectories.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "common_corridors_output")
SEGMENTS_CSV_FOR_CORRIDORS = os.path.join(
    BASE_DIR,
    "common_paths_output",
    "weighted_common_segments.csv"
)

LAT_COL = "LATITUDE"
LON_COL = "LONGITUDE"
TRAJ_COL = "TRAJECTORY_UID"
TIME_COL = "POINT_INDEX"

GRID_SIZE_M = 30

MIN_WEIGHT_TO_DISPLAY = 2
MIN_CORRIDOR_POINTS = 8
MAX_CORRIDORS_TO_DISPLAY = 500
MAP_REGION_FILTER = "idf"
MAP_REGION_BOUNDS = {
    "min_lat": 48.10,
    "max_lat": 49.25,
    "min_lon": 1.45,
    "max_lon": 3.60,
}
MASK_OUTSIDE_MAP_REGION = True
SUPPRESS_OVERLAPPING_CORRIDORS_ON_MAP = True
DISPLAY_OVERLAP_THRESHOLD = 0.35
DISPLAY_OVERLAP_SNAP_DECIMALS = 3
CORRIDOR_RECONSTRUCTION_VERSION = "best_continuation_v2"
ALLOW_INTERSECTION_CONTINUATION = True
MIN_CONTINUATION_COSINE = -0.2
WEIGHT_CONTINUATION_BONUS = 0.35
MAX_CORRIDOR_STEPS = 5000

MIN_SEGMENT_LENGTH_M = 10
MAX_SEGMENT_LENGTH_M = 1000

REUSE_EXISTING_CORRIDORS_CSV = True
USE_EXISTING_SEGMENTS_CSV_FOR_CORRIDORS = True
USE_PARALLEL_AGGREGATION = True
WORKERS = max(1, (os.cpu_count() or 2) - 1)
TRAJECTORIES_PER_TASK = 250

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "weighted_common_corridors.csv")
OUTPUT_MAP = os.path.join(OUTPUT_DIR, "weighted_common_corridors_map.html")
PROFILE_CORRIDORS_CSV = os.path.join(OUTPUT_DIR, "weighted_common_corridors_by_profile.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# FONCTIONS GPS / GRILLE
# =========================

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def latlon_to_grid(lat, lon, grid_size_m, ref_lat):
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * math.cos(math.radians(ref_lat))

    x = lon * meters_per_deg_lon
    y = lat * meters_per_deg_lat

    gx = round(x / grid_size_m)
    gy = round(y / grid_size_m)

    return gx, gy


def grid_to_latlon(cell, grid_size_m, ref_lat):
    gx, gy = cell

    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * math.cos(math.radians(ref_lat))

    lon = (gx * grid_size_m) / meters_per_deg_lon
    lat = (gy * grid_size_m) / meters_per_deg_lat

    return lat, lon


def normalize_edge(cell_a, cell_b):
    return tuple(sorted([cell_a, cell_b]))


def resolve_input_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        return glob.glob(os.path.join(input_path, "*.csv"))

    csv_path = input_path + ".csv"
    if os.path.isfile(csv_path):
        return [csv_path]

    return []


def read_csv_with_needed_columns(file, wanted_columns):
    header = pd.read_csv(file, nrows=0)

    columns = [
        column
        for column in wanted_columns
        if column is not None and column in header.columns
    ]

    if columns:
        return pd.read_csv(file, usecols=columns)

    return pd.read_csv(file)


def count_edges_for_trajectory_records(records, ref_lat):
    edge_counts = Counter()

    for record in records:
        latitudes = record["latitudes"]
        longitudes = record["longitudes"]

        if len(latitudes) < 2:
            continue

        edges_seen_in_this_traj = set()

        for i in range(len(latitudes) - 1):
            lat1 = latitudes[i]
            lon1 = longitudes[i]
            lat2 = latitudes[i + 1]
            lon2 = longitudes[i + 1]

            dist = haversine_m(lat1, lon1, lat2, lon2)

            if dist < MIN_SEGMENT_LENGTH_M or dist > MAX_SEGMENT_LENGTH_M:
                continue

            cell_a = latlon_to_grid(lat1, lon1, GRID_SIZE_M, ref_lat)
            cell_b = latlon_to_grid(lat2, lon2, GRID_SIZE_M, ref_lat)

            if cell_a == cell_b:
                continue

            edge = normalize_edge(cell_a, cell_b)
            edges_seen_in_this_traj.add(edge)

        for edge in edges_seen_in_this_traj:
            edge_counts[edge] += 1

    return edge_counts


def make_trajectory_record(file_name, traj_id, traj_df):
    return {
        "trajectory_id": f"{file_name}__{traj_id}",
        "latitudes": traj_df[LAT_COL].tolist(),
        "longitudes": traj_df[LON_COL].tolist(),
    }


def merge_completed_futures(futures, edge_counts):
    for future in as_completed(futures):
        edge_counts.update(future.result())
    futures.clear()


def aggregate_edges_for_file(file, ref_lat):
    print(f"Traitement : {os.path.basename(file)}")

    try:
        df = read_csv_with_needed_columns(file, [LAT_COL, LON_COL, TRAJ_COL, TIME_COL])
    except Exception as e:
        print("  Erreur lecture :", e)
        return Counter()

    if LAT_COL not in df.columns or LON_COL not in df.columns:
        print("  Colonnes LATITUDE/LONGITUDE absentes.")
        return Counter()

    df = df.dropna(subset=[LAT_COL, LON_COL]).copy()

    df = df[
        (df[LAT_COL].between(-90, 90)) &
        (df[LON_COL].between(-180, 180))
    ]

    if len(df) < 2:
        return Counter()

    if TIME_COL is not None and TIME_COL in df.columns:
        sort_columns = [TIME_COL]
        if TRAJ_COL is not None and TRAJ_COL in df.columns:
            sort_columns = [TRAJ_COL, TIME_COL]
        df = df.sort_values(sort_columns)

    file_name = os.path.basename(file)

    if TRAJ_COL is not None and TRAJ_COL in df.columns:
        groups = df.groupby(TRAJ_COL, sort=False)
    else:
        df["_single_traj"] = file_name
        groups = df.groupby("_single_traj", sort=False)

    if not USE_PARALLEL_AGGREGATION or WORKERS <= 1:
        records = [
            make_trajectory_record(file_name, traj_id, traj_df)
            for traj_id, traj_df in groups
        ]
        return count_edges_for_trajectory_records(records, ref_lat)

    edge_counts = Counter()
    pending_futures = []
    current_records = []
    max_pending_tasks = WORKERS * 2

    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            for traj_id, traj_df in groups:
                current_records.append(make_trajectory_record(file_name, traj_id, traj_df))

                if len(current_records) < TRAJECTORIES_PER_TASK:
                    continue

                pending_futures.append(
                    executor.submit(
                        count_edges_for_trajectory_records,
                        current_records,
                        ref_lat
                    )
                )
                current_records = []

                if len(pending_futures) >= max_pending_tasks:
                    done_future = pending_futures.pop(0)
                    edge_counts.update(done_future.result())

            if current_records:
                pending_futures.append(
                    executor.submit(
                        count_edges_for_trajectory_records,
                        current_records,
                        ref_lat
                    )
                )

            merge_completed_futures(pending_futures, edge_counts)
    except (OSError, PermissionError) as exc:
        print("  Parallele indisponible, fallback sequentiel :", exc)
        records = [
            make_trajectory_record(file_name, traj_id, traj_df)
            for traj_id, traj_df in groups
        ]
        return count_edges_for_trajectory_records(records, ref_lat)

    return edge_counts


# =========================
# RECONSTRUCTION DE CORRIDORS CONTINUS
# =========================

def build_continuous_corridors(edge_weights):
    """
    Transforme un graphe de petits troncons ponderes en lignes continues.

    Ancienne logique : couper a chaque intersection. En zone dense, cela produit
    des micro-segments. Ici on prolonge greedily en choisissant la continuation
    la plus naturelle : direction proche + poids eleve.
    """

    adjacency = defaultdict(list)
    eligible_edges = {}

    for edge, weight in edge_weights.items():
        cell_a, cell_b = edge

        if weight < MIN_WEIGHT_TO_DISPLAY:
            continue

        adjacency[cell_a].append((cell_b, weight))
        adjacency[cell_b].append((cell_a, weight))
        eligible_edges[edge] = weight

    visited_edges = set()
    corridors = []
    max_weight = max(eligible_edges.values()) if eligible_edges else 1

    def edge_key(a, b):
        return normalize_edge(a, b)

    def vector(a, b):
        return b[0] - a[0], b[1] - a[1]

    def cosine_between(a, b, c):
        v1 = vector(a, b)
        v2 = vector(b, c)
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])

        if norm1 == 0 or norm2 == 0:
            return -1

        return ((v1[0] * v2[0]) + (v1[1] * v2[1])) / (norm1 * norm2)

    def choose_next(previous, current):
        candidates = []

        for neighbor, weight in adjacency[current]:
            next_edge = edge_key(current, neighbor)

            if neighbor == previous or next_edge in visited_edges:
                continue

            straightness = cosine_between(previous, current, neighbor)
            score = straightness + WEIGHT_CONTINUATION_BONUS * (weight / max_weight)
            candidates.append((score, straightness, weight, neighbor, next_edge))

        if not candidates:
            return None

        if not ALLOW_INTERSECTION_CONTINUATION and len(candidates) != 1:
            return None

        best = max(candidates, key=lambda item: item[0])

        if best[1] < MIN_CONTINUATION_COSINE:
            return None

        return best

    def extend_forward(path, weights):
        while len(path) < MAX_CORRIDOR_STEPS:
            previous = path[-2]
            current = path[-1]
            next_choice = choose_next(previous, current)

            if next_choice is None:
                break

            _, _, weight, neighbor, next_edge = next_choice
            visited_edges.add(next_edge)
            path.append(neighbor)
            weights.append(weight)

    def trace_from_edge(cell_a, cell_b):
        first_edge = edge_key(cell_a, cell_b)

        if first_edge in visited_edges:
            return [], []

        visited_edges.add(first_edge)
        path = [cell_a, cell_b]
        weights = [eligible_edges[first_edge]]

        extend_forward(path, weights)

        path.reverse()
        extend_forward(path, weights)
        path.reverse()

        return path, weights

    for edge, weight in sorted(eligible_edges.items(), key=lambda item: item[1], reverse=True):
        if edge in visited_edges:
            continue

        cell_a, cell_b = edge
        path, weights = trace_from_edge(cell_a, cell_b)

        if len(path) < MIN_CORRIDOR_POINTS or not weights:
            continue

        weight_mean = sum(weights) / len(weights)
        length_cells = len(path)
        corridors.append({
            "cells": path,
            "weight_mean": weight_mean,
            "weight_max": max(weights),
            "weight_min": min(weights),
            "length_cells": length_cells,
            "display_score": weight_mean * math.log1p(length_cells)
        })

    corridors = sorted(
        corridors,
        key=lambda c: (c["display_score"], c["weight_mean"], c["length_cells"]),
        reverse=True
    )

    return corridors


def corridors_to_dataframe(corridors, ref_lat):
    rows = []

    for corridor_id, corridor in enumerate(corridors, start=1):
        coords = [
            grid_to_latlon(cell, GRID_SIZE_M, ref_lat)
            for cell in corridor["cells"]
        ]

        latitudes = [lat for lat, lon in coords]
        longitudes = [lon for lat, lon in coords]

        rows.append({
            "corridor_id": corridor_id,
            "weight_mean": round(corridor["weight_mean"], 2),
            "weight_max": corridor["weight_max"],
            "weight_min": corridor["weight_min"],
            "length_points": len(coords),
            "display_score": round(corridor["display_score"], 2),
            "reconstruction_version": CORRIDOR_RECONSTRUCTION_VERSION,
            "latitudes": ";".join(map(str, latitudes)),
            "longitudes": ";".join(map(str, longitudes))
        })

    return pd.DataFrame(rows)


def coords_length_km(coords):
    if len(coords) < 2:
        return 0

    total_m = 0

    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]
        total_m += haversine_m(lat1, lon1, lat2, lon2)

    return total_m / 1000


def direct_distance_km(coords):
    if len(coords) < 2:
        return 0

    lat1, lon1 = coords[0]
    lat2, lon2 = coords[-1]
    return haversine_m(lat1, lon1, lat2, lon2) / 1000


def ensure_corridor_metrics(corridors_df):
    if corridors_df.empty:
        return corridors_df

    corridors_df = corridors_df.copy()

    missing_metric_columns = [
        column
        for column in ["length_km", "direct_distance_km", "sinuosity"]
        if column not in corridors_df.columns
    ]

    if missing_metric_columns:
        lengths = []
        direct_distances = []
        sinuosities = []

        for _, row in corridors_df.iterrows():
            coords = parse_corridor_coordinates(row)
            length_km = coords_length_km(coords)
            distance_km = direct_distance_km(coords)
            sinuosity = length_km / distance_km if distance_km > 0 else 1

            lengths.append(round(length_km, 3))
            direct_distances.append(round(distance_km, 3))
            sinuosities.append(round(sinuosity, 3))

        corridors_df["length_km"] = lengths
        corridors_df["direct_distance_km"] = direct_distances
        corridors_df["sinuosity"] = sinuosities

    if "display_score" not in corridors_df.columns:
        corridors_df["display_score"] = (
            corridors_df["weight_mean"] * corridors_df["length_points"].apply(math.log1p)
        ).round(2)

    return corridors_df


def coordinate_corridors_to_dataframe(corridors):
    rows = []

    for corridor_id, corridor in enumerate(corridors, start=1):
        coords = corridor["cells"]
        latitudes = [lat for lat, lon in coords]
        longitudes = [lon for lat, lon in coords]

        rows.append({
            "corridor_id": corridor_id,
            "weight_mean": round(corridor["weight_mean"], 2),
            "weight_max": corridor["weight_max"],
            "weight_min": corridor["weight_min"],
            "length_points": len(coords),
            "display_score": round(corridor["display_score"], 2),
            "reconstruction_version": CORRIDOR_RECONSTRUCTION_VERSION,
            "latitudes": ";".join(map(str, latitudes)),
            "longitudes": ";".join(map(str, longitudes))
        })

    return pd.DataFrame(rows)


def load_edge_weights_from_segments_csv(segments_csv):
    segments_df = pd.read_csv(segments_csv)
    required_columns = {"lat1", "lon1", "lat2", "lon2", "weight"}
    missing_columns = required_columns - set(segments_df.columns)

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le CSV de segments : "
            + ", ".join(sorted(missing_columns))
        )

    edge_weights = Counter()

    for row in segments_df.itertuples(index=False):
        cell_a = (round(float(row.lat1), 8), round(float(row.lon1), 8))
        cell_b = (round(float(row.lat2), 8), round(float(row.lon2), 8))

        if cell_a == cell_b:
            continue

        edge = normalize_edge(cell_a, cell_b)
        edge_weights[edge] = max(edge_weights[edge], int(row.weight))

    return edge_weights


def build_corridors_dataframe_from_edge_weights(
    edge_weights,
    ref_lat=None,
    from_coordinates=False
):
    print("Connexions internes generees :", len(edge_weights))

    corridors = build_continuous_corridors(edge_weights)

    print("Corridors continus generes :", len(corridors))

    if not corridors:
        raise ValueError(
            "Aucun corridor genere. Essaie de baisser MIN_WEIGHT_TO_DISPLAY "
            "ou d'augmenter GRID_SIZE_M."
        )

    if from_coordinates:
        return ensure_corridor_metrics(coordinate_corridors_to_dataframe(corridors))

    if ref_lat is None:
        raise ValueError("ref_lat est requis pour convertir des cellules de grille.")

    return ensure_corridor_metrics(corridors_to_dataframe(corridors, ref_lat))


def parse_corridor_coordinates(row):
    latitudes = [float(x) for x in str(row["latitudes"]).split(";") if x]
    longitudes = [float(x) for x in str(row["longitudes"]).split(";") if x]

    return list(zip(latitudes, longitudes))


def is_inside_map_region_bounds(lat, lon):
    return (
        MAP_REGION_BOUNDS["min_lat"] <= lat <= MAP_REGION_BOUNDS["max_lat"]
        and MAP_REGION_BOUNDS["min_lon"] <= lon <= MAP_REGION_BOUNDS["max_lon"]
    )


def split_coords_inside_map_region(coords):
    if not MAP_REGION_FILTER:
        return [coords] if len(coords) >= 2 else []

    parts = []
    current_part = []

    for lat, lon in coords:
        if is_inside_map_region_bounds(lat, lon):
            current_part.append((lat, lon))
            continue

        if len(current_part) >= 2:
            parts.append(current_part)

        current_part = []

    if len(current_part) >= 2:
        parts.append(current_part)

    return parts


def corridor_has_map_region_points(row):
    coords = parse_corridor_coordinates(row)
    return any(is_inside_map_region_bounds(lat, lon) for lat, lon in coords)


def add_region_mask(map_object):
    if not MASK_OUTSIDE_MAP_REGION or not MAP_REGION_FILTER:
        return

    outer_bounds = [
        [-90, -180],
        [-90, 180],
        [90, 180],
        [90, -180],
    ]
    inner_bounds = [
        [MAP_REGION_BOUNDS["min_lat"], MAP_REGION_BOUNDS["min_lon"]],
        [MAP_REGION_BOUNDS["max_lat"], MAP_REGION_BOUNDS["min_lon"]],
        [MAP_REGION_BOUNDS["max_lat"], MAP_REGION_BOUNDS["max_lon"]],
        [MAP_REGION_BOUNDS["min_lat"], MAP_REGION_BOUNDS["max_lon"]],
    ]

    folium.Polygon(
        locations=[outer_bounds, inner_bounds],
        color="white",
        weight=0,
        fill=True,
        fill_color="white",
        fill_opacity=0.92,
        interactive=False,
    ).add_to(map_object)

    folium.Rectangle(
        bounds=[
            [MAP_REGION_BOUNDS["min_lat"], MAP_REGION_BOUNDS["min_lon"]],
            [MAP_REGION_BOUNDS["max_lat"], MAP_REGION_BOUNDS["max_lon"]],
        ],
        color="#111827",
        weight=1,
        fill=False,
        opacity=0.25,
        interactive=False,
    ).add_to(map_object)


def corridor_display_cells(coords):
    return {
        (
            round(lat, DISPLAY_OVERLAP_SNAP_DECIMALS),
            round(lon, DISPLAY_OVERLAP_SNAP_DECIMALS)
        )
        for lat, lon in coords
    }


def filter_overlapping_corridors_for_map(corridors_df):
    if MAP_REGION_FILTER:
        corridors_df = corridors_df[corridors_df.apply(corridor_has_map_region_points, axis=1)]
        print(f"Corridors intersectant {MAP_REGION_FILTER} :", len(corridors_df))

    if not SUPPRESS_OVERLAPPING_CORRIDORS_ON_MAP:
        return corridors_df.head(MAX_CORRIDORS_TO_DISPLAY).copy()

    selected_rows = []
    occupied_cells = set()
    skipped_overlap = 0

    for _, row in corridors_df.iterrows():
        coords = parse_corridor_coordinates(row)

        if len(coords) < 2:
            continue

        cells = corridor_display_cells(coords)

        if not cells:
            continue

        overlap_ratio = len(cells & occupied_cells) / len(cells)

        if overlap_ratio > DISPLAY_OVERLAP_THRESHOLD:
            skipped_overlap += 1
            continue

        selected_rows.append(row)
        occupied_cells.update(cells)

        if len(selected_rows) >= MAX_CORRIDORS_TO_DISPLAY:
            break

    print("Corridors masques car trop superposes :", skipped_overlap)

    if not selected_rows:
        return corridors_df.head(MAX_CORRIDORS_TO_DISPLAY).copy()

    return pd.DataFrame(selected_rows)


def build_map_from_corridors(corridors_df):
    if corridors_df.empty:
        raise ValueError("Aucun corridor Ã  afficher.")

    required_columns = {
        "corridor_id",
        "weight_mean",
        "weight_max",
        "weight_min",
        "latitudes",
        "longitudes"
    }

    missing = required_columns - set(corridors_df.columns)

    if missing:
        raise ValueError(
            "Colonnes manquantes dans le CSV de corridors : "
            + ", ".join(sorted(missing))
        )

    corridors_df = ensure_corridor_metrics(corridors_df)

    if MAP_REGION_FILTER:
        center_lat = (MAP_REGION_BOUNDS["min_lat"] + MAP_REGION_BOUNDS["max_lat"]) / 2
        center_lon = (MAP_REGION_BOUNDS["min_lon"] + MAP_REGION_BOUNDS["max_lon"]) / 2
        zoom_start = 10
    else:
        all_lat = []
        all_lon = []

        for _, row in corridors_df.iterrows():
            coords = parse_corridor_coordinates(row)

            for lat, lon in coords:
                all_lat.append(lat)
                all_lon.append(lon)

        center_lat = sum(all_lat) / len(all_lat)
        center_lon = sum(all_lon) / len(all_lon)
        zoom_start = 11

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="cartodbpositron",
        prefer_canvas=True
    )

    if MAP_REGION_FILTER:
        m.fit_bounds([
            [MAP_REGION_BOUNDS["min_lat"], MAP_REGION_BOUNDS["min_lon"]],
            [MAP_REGION_BOUNDS["max_lat"], MAP_REGION_BOUNDS["max_lon"]],
        ])
        m.options["maxBounds"] = [
            [MAP_REGION_BOUNDS["min_lat"], MAP_REGION_BOUNDS["min_lon"]],
            [MAP_REGION_BOUNDS["max_lat"], MAP_REGION_BOUNDS["max_lon"]],
        ]
        m.options["maxBoundsViscosity"] = 1.0

    Fullscreen().add_to(m)
    MiniMap().add_to(m)
    MeasureControl().add_to(m)
    add_region_mask(m)

    corridors_df = corridors_df.sort_values(
        ["display_score", "weight_mean", "length_points"],
        ascending=False
    )
    display_df = filter_overlapping_corridors_for_map(corridors_df)

    max_weight = display_df["weight_mean"].max()
    max_length = display_df["length_km"].max()
    corridor_layer_data = []

    print("Corridors affichÃ©s :", len(display_df), "/", len(corridors_df))

    for _, row in display_df.iterrows():
        coords = parse_corridor_coordinates(row)
        coord_parts = split_coords_inside_map_region(coords)

        if not coord_parts:
            continue

        weight = row["weight_mean"]
        line_width = 2 + 10 * (weight / max_weight)

        if weight >= max_weight * 0.66:
            color = "red"
        elif weight >= max_weight * 0.33:
            color = "orange"
        else:
            color = "blue"

        for coord_part in coord_parts:
            polyline = folium.PolyLine(
                locations=coord_part,
                weight=line_width,
                color=color,
                opacity=0.75,
                tooltip=(
                    f"Corridor {int(row['corridor_id'])} | "
                    f"Poids moyen : {round(weight, 2)} | "
                    f"Poids max : {int(row['weight_max'])}"
                )
            )
            polyline.add_to(m)
            corridor_layer_data.append({
                "layer": polyline.get_name(),
                "corridor_id": int(row["corridor_id"]),
                "weight_mean": float(weight),
                "weight_max": int(row["weight_max"]),
                "length_km": float(row["length_km"]),
                "sinuosity": float(row["sinuosity"]),
                "intensity": color,
            })

    profile_filter_note = (
        "Profil non disponible dans ce CSV"
        if not os.path.exists(PROFILE_CORRIDORS_CSV)
        else "Profil CSV detecte"
    )
    legend_html = f"""
<div style="
    position: fixed;
    top: 72px;
    left: 12px;
    z-index: 9999;
    background: white;
    padding: 10px 12px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.16);
    font-family: Arial, sans-serif;
    color: #111827;
    font-size: 11px;
    line-height: 1.35;
    width: 292px;
">
    <div style="font-weight:700; margin-bottom:7px;">Intensite des couloirs</div>
    <div style="display:flex; align-items:center; gap:8px; margin:5px 0;">
        <span style="display:inline-block;width:22px;height:4px;background:red;border-radius:2px;"></span>
        <span>Fort : >= 66% du max affiche</span>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin:5px 0;">
        <span style="display:inline-block;width:22px;height:4px;background:orange;border-radius:2px;"></span>
        <span>Moyen : 33% a 66%</span>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin:5px 0;">
        <span style="display:inline-block;width:22px;height:4px;background:blue;border-radius:2px;"></span>
        <span>Faible : moins de 33%</span>
    </div>
    <div style="margin-top:7px;color:#4b5563;">
        weight_mean = poids moyen des segments du couloir
    </div>
    <div style="height:1px;background:#e5e7eb;margin:10px 0;"></div>
    <label style="display:block;font-weight:700;margin-bottom:4px;">Poids moyen minimum</label>
    <input id="weight-filter" type="range" min="0" max="{round(float(max_weight), 2)}" value="0" step="1" style="width:100%;">
    <div style="display:flex;justify-content:space-between;color:#4b5563;">
        <span>0</span><span id="weight-filter-value">0</span>
    </div>
    <label style="display:block;font-weight:700;margin:8px 0 4px;">Longueur minimum (km)</label>
    <input id="length-filter" type="range" min="0" max="{round(float(max_length), 2)}" value="0" step="0.5" style="width:100%;">
    <div style="display:flex;justify-content:space-between;color:#4b5563;">
        <span>0</span><span id="length-filter-value">0 km</span>
    </div>
    <label style="display:block;font-weight:700;margin:8px 0 4px;">Intensite</label>
    <select id="intensity-filter" style="width:100%;font-size:11px;padding:4px;">
        <option value="all">Toutes</option>
        <option value="red">Fort</option>
        <option value="orange">Moyen</option>
        <option value="blue">Faible</option>
    </select>
    <label style="display:block;font-weight:700;margin:8px 0 4px;">Profil</label>
    <select id="profile-filter" disabled style="width:100%;font-size:11px;padding:4px;color:#6b7280;">
        <option>{profile_filter_note}</option>
    </select>
    <div id="corridor-count" style="margin-top:8px;color:#111827;font-weight:700;"></div>
</div>
"""
    m.get_root().html.add_child(Element(legend_html))

    map_name = m.get_name()
    controls_js = f"""
(function() {{
  const MAP_NAME = "{map_name}";
  const corridors = {json.dumps(corridor_layer_data)};
  const maxAttempts = 200;
  let attempts = 0;
  let filtersReady = false;

  function getMap() {{
    return window[MAP_NAME] || null;
  }}

  function layerFor(item) {{
    if (window[item.layer]) return window[item.layer];
    try {{
      return eval(item.layer);
    }} catch (error) {{
      return null;
    }}
  }}

  function bootFilters() {{
    if (filtersReady) return;

    const map = getMap();
    const weightInput = document.getElementById("weight-filter");
    const lengthInput = document.getElementById("length-filter");
    const intensityInput = document.getElementById("intensity-filter");
    const weightValue = document.getElementById("weight-filter-value");
    const lengthValue = document.getElementById("length-filter-value");
    const countLabel = document.getElementById("corridor-count");
    const layersReady = corridors.length === 0 || corridors.some(function(item) {{
      return !!layerFor(item);
    }});

    if (!map || !weightInput || !lengthInput || !intensityInput || !layersReady) {{
      attempts += 1;
      if (attempts < maxAttempts) {{
        window.setTimeout(bootFilters, 50);
      }} else if (countLabel) {{
        countLabel.textContent = "Filtres non initialises";
      }}
      return;
    }}

    filtersReady = true;

    function applyFilters() {{
      const minWeight = parseFloat(weightInput.value || "0");
      const minLength = parseFloat(lengthInput.value || "0");
      const intensity = intensityInput.value;
      let shown = 0;

      weightValue.textContent = minWeight.toFixed(0);
      lengthValue.textContent = minLength.toFixed(1) + " km";

      corridors.forEach(function(item) {{
        const layer = layerFor(item);
        if (!layer) return;

        const visible =
          item.weight_mean >= minWeight &&
          item.length_km >= minLength &&
          (intensity === "all" || item.intensity === intensity);

        if (visible) {{
          if (!map.hasLayer(layer)) layer.addTo(map);
          shown += 1;
        }} else if (map.hasLayer(layer)) {{
          map.removeLayer(layer);
        }}
      }});

      countLabel.textContent = shown + " corridors affiches";
    }}

    weightInput.addEventListener("input", applyFilters);
    lengthInput.addEventListener("input", applyFilters);
    intensityInput.addEventListener("change", applyFilters);
    applyFilters();
  }}

  window.setTimeout(bootFilters, 0);
}})();
"""
    m.get_root().script.add_child(Element(controls_js))

    m.save(OUTPUT_MAP)

    print("Carte exportÃ©e :", OUTPUT_MAP)


def main():
    # =========================
    # REUTILISATION SI CSV EXISTE
    # =========================

    if REUSE_EXISTING_CORRIDORS_CSV and os.path.exists(OUTPUT_CSV):
        existing_df = ensure_corridor_metrics(pd.read_csv(OUTPUT_CSV))
        existing_version = ""

        if "reconstruction_version" in existing_df.columns and not existing_df.empty:
            existing_version = str(existing_df["reconstruction_version"].iloc[0])

        if existing_version == CORRIDOR_RECONSTRUCTION_VERSION:
            print("CSV de corridors existant trouve :", OUTPUT_CSV)
            print("Generation de la carte uniquement.")
            existing_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            build_map_from_corridors(existing_df)
            return

        print("CSV de corridors existant obsolete :", OUTPUT_CSV)
        print("Reconstruction avec la nouvelle logique de lignes continues.")

    if (
        USE_EXISTING_SEGMENTS_CSV_FOR_CORRIDORS
        and os.path.exists(SEGMENTS_CSV_FOR_CORRIDORS)
    ):
        print("CSV de segments existant trouve :", SEGMENTS_CSV_FOR_CORRIDORS)
        print("Construction des corridors depuis les segments, sans relire les GPS.")
        edge_weights = load_edge_weights_from_segments_csv(SEGMENTS_CSV_FOR_CORRIDORS)
        corridors_df = build_corridors_dataframe_from_edge_weights(
            edge_weights,
            from_coordinates=True
        )
        corridors_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print("CSV corridors exporte :", OUTPUT_CSV)
        print("Poids moyen max :", corridors_df["weight_mean"].max())
        print("Poids max global :", corridors_df["weight_max"].max())
        build_map_from_corridors(corridors_df)
        return

    # =========================
    # CHARGEMENT DES DONNEES
    # =========================

    files = resolve_input_files(INPUT_PATH)

    print("Fichiers trouves :", len(files))

    if not files:
        raise FileNotFoundError(
            "Aucun fichier CSV trouve. Verifie INPUT_PATH : "
            f"{INPUT_PATH}"
        )

    lat_sum = 0
    lat_count = 0

    for file in files:
        try:
            df_tmp = pd.read_csv(file, usecols=[LAT_COL])
            valid_lats = df_tmp[LAT_COL].dropna()
            lat_sum += valid_lats.sum()
            lat_count += len(valid_lats)
        except Exception:
            pass

    if lat_count == 0:
        raise ValueError("Impossible de trouver des latitudes valides.")

    ref_lat = lat_sum / lat_count

    print("Latitude de reference :", ref_lat)

    # =========================
    # AGREGATION INTERNE
    # =========================

    print("Parallele active :", USE_PARALLEL_AGGREGATION and WORKERS > 1)
    print("Workers :", WORKERS)
    print("Trajectoires par tache :", TRAJECTORIES_PER_TASK)

    edge_weights = Counter()

    for file_idx, file in enumerate(files, start=1):
        print(f"[{file_idx}/{len(files)}]")
        edge_weights.update(aggregate_edges_for_file(file, ref_lat))

    # =========================
    # CONSTRUCTION DES CORRIDORS
    # =========================

    corridors_df = build_corridors_dataframe_from_edge_weights(
        edge_weights,
        ref_lat=ref_lat
    )

    corridors_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("CSV exporte :", OUTPUT_CSV)
    print("Poids moyen max :", corridors_df["weight_mean"].max())
    print("Poids max global :", corridors_df["weight_max"].max())

    # =========================
    # CARTE
    # =========================

    build_map_from_corridors(corridors_df)


if __name__ == "__main__":
    main()
