import os
import math
import time
import glob
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import folium
from tqdm import tqdm
from folium.plugins import Fullscreen, MiniMap, MousePosition, MeasureControl
from branca.element import Template, MacroElement

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(
    BASE_DIR,
    "NetMob25CleanedData",
    "NetMob25CleanedData",
    "gps_dataset"
)
CSV_NAME = "12_3332.csv"

OUTPUT_FILE = "map_12_3332_segmented_pipeline.html"
BATCH_OUTPUT_FILE = "all_refined_trajectories.csv"
BATCH_SUMMARY_FILE = "batch_processing_summary.csv"
RANDOM_MAP_OUTPUT_FILE = "map_random_50_users.html"
RANDOM_MAP_SUMMARY_FILE = "random_map_summary.csv"
DEFAULT_WORKERS = 8
DEFAULT_RANDOM_MAP_USERS = 50

STAYPOINT_TIME_MIN = 10
STAYPOINT_RADIUS_M = 50
MERGE_STAYPOINT_RADIUS_M = 100

LOCAL_RADIUS_M = 100
SAMPLE_STEP = 5
MIN_POINTS_TRAJ = 10

PRE_SAMPLE_MAX_DISTANCE_M = 2
PRE_SAMPLE_MAX_TIME_SECONDS = 2

IDF_BOUNDS = dict(min_lat=48.10, max_lat=49.25, min_lon=1.40, max_lon=3.60)

COLORS = [
    "#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316",
    "#14b8a6", "#ec4899", "#84cc16", "#6366f1", "#facc15"
]


def haversine(lat1, lon1, lat2, lon2):
    earth_radius_m = 6_371_000.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    return 2 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)

    data = data.dropna(subset=["LATITUDE", "LONGITUDE"])

    data = data[
        (data["LATITUDE"].between(-90, 90)) &
        (data["LONGITUDE"].between(-180, 180))
    ]

    data = data[
        (data["LATITUDE"].between(IDF_BOUNDS["min_lat"], IDF_BOUNDS["max_lat"])) &
        (data["LONGITUDE"].between(IDF_BOUNDS["min_lon"], IDF_BOUNDS["max_lon"]))
    ]

    if "UTC_DATE" in data.columns and "UTC_TIME" in data.columns:
        data["DATETIME"] = pd.to_datetime(
            data["UTC_DATE"].astype(str) + " " + data["UTC_TIME"].astype(str),
            errors="coerce"
        )
    elif "LOCAL_DATE" in data.columns and "LOCAL_TIME" in data.columns:
        data["DATETIME"] = pd.to_datetime(
            data["LOCAL_DATE"].astype(str) + " " + data["LOCAL_TIME"].astype(str),
            errors="coerce"
        )
    else:
        raise ValueError("Colonnes de temps manquantes.")

    data = data.dropna(subset=["DATETIME"])
    data = data.sort_values("DATETIME").reset_index(drop=True)

    return data


def light_presampling(data, max_distance_m=10, max_time_seconds=10, show_progress=True):
    """
    Pré-sampling léger :
    fusionne les points consécutifs très proches dans l'espace et dans le temps.
    """

    if len(data) < 2:
        return data.copy()

    reduced_points = []
    buffer_points = [data.iloc[0]]

    for i in tqdm(
        range(1, len(data)),
        desc="Pré-sampling léger",
        unit="pt",
        disable=not show_progress
    ):
        previous_point = buffer_points[-1]
        current_point = data.iloc[i]

        distance = haversine(
            previous_point["LATITUDE"],
            previous_point["LONGITUDE"],
            current_point["LATITUDE"],
            current_point["LONGITUDE"]
        )

        time_gap = (
            current_point["DATETIME"] - previous_point["DATETIME"]
        ).total_seconds()

        if distance <= max_distance_m and time_gap <= max_time_seconds:
            buffer_points.append(current_point)
        else:
            buffer_df = pd.DataFrame(buffer_points)

            averaged_point = buffer_df.iloc[-1].copy()
            averaged_point["LATITUDE"] = buffer_df["LATITUDE"].mean()
            averaged_point["LONGITUDE"] = buffer_df["LONGITUDE"].mean()
            averaged_point["DATETIME"] = buffer_df["DATETIME"].iloc[-1]

            reduced_points.append(averaged_point)

            buffer_points = [current_point]

    if buffer_points:
        buffer_df = pd.DataFrame(buffer_points)

        averaged_point = buffer_df.iloc[-1].copy()
        averaged_point["LATITUDE"] = buffer_df["LATITUDE"].mean()
        averaged_point["LONGITUDE"] = buffer_df["LONGITUDE"].mean()
        averaged_point["DATETIME"] = buffer_df["DATETIME"].iloc[-1]

        reduced_points.append(averaged_point)

    return pd.DataFrame(reduced_points).reset_index(drop=True)


# =========================
# SEGMENTATION ANCIENNE
# =========================

def segment_trajectories_by_staypoints(data, show_progress=True):
    """
    Ancienne segmentation :
    si l'utilisateur reste dans un rayon de STAYPOINT_RADIUS_M
    pendant au moins STAYPOINT_TIME_MIN minutes, on détecte un staypoint.

    Le staypoint marque la fin du trajet courant.
    """

    trajectory_segments = []
    detected_staypoints = []
    current_segment = []

    i = 0
    total_points = len(data)

    progress_bar = tqdm(
        total=total_points,
        desc="Segmentation spatialo-temporelle",
        unit="pt",
        disable=not show_progress
    )

    while i < total_points:
        previous_i = i
        current_segment.append(data.iloc[i])

        j = i + 1
        staypoint_detected = False

        while j < total_points:
            distance = haversine(
                data.iloc[i]["LATITUDE"],
                data.iloc[i]["LONGITUDE"],
                data.iloc[j]["LATITUDE"],
                data.iloc[j]["LONGITUDE"]
            )

            duration = (
                data.iloc[j]["DATETIME"] - data.iloc[i]["DATETIME"]
            ).total_seconds()

            if distance > STAYPOINT_RADIUS_M:
                break

            if duration >= STAYPOINT_TIME_MIN * 60:
                staypoint_detected = True
                break

            j += 1

        if staypoint_detected:
            stay_lat = data.iloc[i:j + 1]["LATITUDE"].mean()
            stay_lon = data.iloc[i:j + 1]["LONGITUDE"].mean()

            detected_staypoints.append({
                "latitude": stay_lat,
                "longitude": stay_lon,
                "start_time": data.iloc[i]["DATETIME"],
                "end_time": data.iloc[j]["DATETIME"],
                "duration_min": duration / 60
            })

            segment_df = pd.DataFrame(current_segment)

            if len(segment_df) >= MIN_POINTS_TRAJ:
                trajectory_segments.append(segment_df.reset_index(drop=True))

            current_segment = []
            i = j + 1
        else:
            i += 1

        progress_bar.update(max(1, i - previous_i))

    progress_bar.close()

    if current_segment:
        segment_df = pd.DataFrame(current_segment)

        if len(segment_df) >= MIN_POINTS_TRAJ:
            trajectory_segments.append(segment_df.reset_index(drop=True))

    return trajectory_segments, detected_staypoints


def merge_staypoints(staypoints, threshold_m):
    """
    Regroupe les staypoints proches afin d'éviter de compter plusieurs fois
    le même lieu réel.
    """

    merged_staypoints = []

    for staypoint in staypoints:
        lat = staypoint["latitude"]
        lon = staypoint["longitude"]

        matched_cluster = None

        for cluster in merged_staypoints:
            distance = haversine(
                lat,
                lon,
                cluster["latitude"],
                cluster["longitude"]
            )

            if distance <= threshold_m:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            merged_staypoints.append({
                "latitude": lat,
                "longitude": lon,
                "count": 1
            })
        else:
            count = matched_cluster["count"]

            matched_cluster["latitude"] = (
                matched_cluster["latitude"] * count + lat
            ) / (count + 1)

            matched_cluster["longitude"] = (
                matched_cluster["longitude"] * count + lon
            ) / (count + 1)

            matched_cluster["count"] += 1

    return merged_staypoints


# =========================
# TRIMMING
# =========================

def trim_start_end_zones(trajectory, radius_m):
    if len(trajectory) < 2:
        return trajectory.copy()

    trajectory = trajectory.copy().reset_index(drop=True)

    start_point = trajectory.iloc[0]
    end_point = trajectory.iloc[-1]

    trajectory["distance_from_start"] = trajectory.apply(
        lambda row: haversine(
            start_point["LATITUDE"],
            start_point["LONGITUDE"],
            row["LATITUDE"],
            row["LONGITUDE"]
        ),
        axis=1
    )

    trajectory["distance_from_end"] = trajectory.apply(
        lambda row: haversine(
            end_point["LATITUDE"],
            end_point["LONGITUDE"],
            row["LATITUDE"],
            row["LONGITUDE"]
        ),
        axis=1
    )

    trimmed_trajectory = trajectory[
        (trajectory["distance_from_start"] > radius_m) &
        (trajectory["distance_from_end"] > radius_m)
    ]

    return trimmed_trajectory.reset_index(drop=True)


# =========================
# MAP
# =========================

def build_map(all_points, processed_trajectories, merged_staypoints, show_progress=True):
    center = [
        all_points["LATITUDE"].mean(),
        all_points["LONGITUDE"].mean()
    ]

    map_object = folium.Map(
        location=center,
        zoom_start=13,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer("CartoDB positron", name="Fond clair").add_to(map_object)
    folium.TileLayer("CartoDB dark_matter", name="Fond sombre").add_to(map_object)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(map_object)

    raw_points = all_points[["LATITUDE", "LONGITUDE"]].values.tolist()

    folium.PolyLine(
        raw_points,
        color="#ef4444",
        weight=3,
        opacity=0.18,
        tooltip="CSV brut complet"
    ).add_to(map_object)

    for index, trajectory_data in enumerate(
        tqdm(
            processed_trajectories,
            desc="Ajout des trajets à la carte",
            unit="trajet",
            disable=not show_progress
        )
    ):
        color = COLORS[index % len(COLORS)]

        raw_trajectory = trajectory_data["raw"]
        trimmed_trajectory = trajectory_data["trimmed"]
        sampled_trajectory = trajectory_data["sampled"]

        if len(trimmed_trajectory) >= 2:
            folium.PolyLine(
                trimmed_trajectory[["LATITUDE", "LONGITUDE"]].values.tolist(),
                color=color,
                weight=4,
                opacity=0.9,
                tooltip=f"Trajet {index + 1} après trimming"
            ).add_to(map_object)

        for _, row in sampled_trajectory.iterrows():
            folium.CircleMarker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                tooltip=f"Point samplé trajet {index + 1}"
            ).add_to(map_object)

        if len(raw_trajectory) >= 2:
            raw_start = raw_trajectory.iloc[0]
            raw_end = raw_trajectory.iloc[-1]

            folium.Circle(
                location=[raw_start["LATITUDE"], raw_start["LONGITUDE"]],
                radius=LOCAL_RADIUS_M,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.07,
                tooltip=f"Zone départ trajet {index + 1}"
            ).add_to(map_object)

            folium.Circle(
                location=[raw_end["LATITUDE"], raw_end["LONGITUDE"]],
                radius=LOCAL_RADIUS_M,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.07,
                tooltip=f"Zone arrivée trajet {index + 1}"
            ).add_to(map_object)

            folium.CircleMarker(
                location=[raw_start["LATITUDE"], raw_start["LONGITUDE"]],
                radius=7,
                color="#16a34a",
                fill=True,
                fill_color="#22c55e",
                fill_opacity=1,
                tooltip=f"Départ brut trajet {index + 1}"
            ).add_to(map_object)

            folium.CircleMarker(
                location=[raw_end["LATITUDE"], raw_end["LONGITUDE"]],
                radius=7,
                color="#991b1b",
                fill=True,
                fill_color="#ef4444",
                fill_opacity=1,
                tooltip=f"Arrivée brute trajet {index + 1}"
            ).add_to(map_object)

        if len(trimmed_trajectory) >= 2:
            trimmed_start = trimmed_trajectory.iloc[0]
            trimmed_end = trimmed_trajectory.iloc[-1]

            folium.CircleMarker(
                location=[trimmed_start["LATITUDE"], trimmed_start["LONGITUDE"]],
                radius=5,
                color="#1d4ed8",
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=1,
                tooltip=f"Premier point conservé trajet {index + 1}"
            ).add_to(map_object)

            folium.CircleMarker(
                location=[trimmed_end["LATITUDE"], trimmed_end["LONGITUDE"]],
                radius=5,
                color="#111827",
                fill=True,
                fill_color="#374151",
                fill_opacity=1,
                tooltip=f"Dernier point conservé trajet {index + 1}"
            ).add_to(map_object)

    for index, staypoint in enumerate(merged_staypoints):
        folium.CircleMarker(
            location=[staypoint["latitude"], staypoint["longitude"]],
            radius=9,
            color="#000000",
            fill=True,
            fill_color="#facc15",
            fill_opacity=1,
            tooltip=f"Lieu unique {index + 1} — {staypoint['count']} staypoints regroupés"
        ).add_to(map_object)

        folium.Circle(
            location=[staypoint["latitude"], staypoint["longitude"]],
            radius=MERGE_STAYPOINT_RADIUS_M,
            color="#000000",
            weight=1,
            fill=True,
            fill_color="#facc15",
            fill_opacity=0.08,
            tooltip=f"Zone de regroupement lieu unique {index + 1}"
        ).add_to(map_object)

    Fullscreen().add_to(map_object)
    MiniMap(toggle_display=True, position="bottomright").add_to(map_object)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordonnées",
        num_digits=5
    ).add_to(map_object)
    MeasureControl().add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)

    map_object.fit_bounds([
        [all_points["LATITUDE"].min(), all_points["LONGITUDE"].min()],
        [all_points["LATITUDE"].max(), all_points["LONGITUDE"].max()]
    ])

    return map_object


def inject_info_panel(
    map_object,
    all_points,
    trajectory_segments,
    processed_trajectories,
    staypoints,
    merged_staypoints
):
    total_trimmed_points = sum(len(t["trimmed"]) for t in processed_trajectories)
    total_sampled_points = sum(len(t["sampled"]) for t in processed_trajectories)

    panel_html = f"""
{{% macro html(this, kwargs) %}}

<style>
#gps-panel {{
    position: fixed;
    top: 20px;
    left: 20px;
    width: 350px;
    z-index: 9999;
    background: white;
    border-radius: 16px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    padding: 18px;
    font-family: Arial, sans-serif;
    color: #0f172a;
}}

#gps-panel h2 {{
    margin: 0 0 10px;
    font-size: 18px;
}}

#gps-panel p {{
    margin: 6px 0;
    font-size: 13px;
    line-height: 1.45;
    color: #475569;
}}

.stat {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
}}

.card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px;
}}

.value {{
    font-weight: bold;
    font-size: 18px;
    color: #2563eb;
}}

.label {{
    font-size: 11px;
    color: #64748b;
    margin-top: 2px;
}}
</style>

<div id="gps-panel">
    <h2>Trajectoires GPS — {CSV_NAME}</h2>

    <p>
        Pipeline : nettoyage → staypoints spatialo-temporels →
        segmentation → regroupement des lieux → trimming → sampling.
    </p>

    <div class="stat">
        <div class="card">
            <div class="value">{len(all_points)}</div>
            <div class="label">points bruts</div>
        </div>

        <div class="card">
            <div class="value">{len(trajectory_segments)}</div>
            <div class="label">trajets détectés</div>
        </div>

        <div class="card">
            <div class="value">{len(staypoints)}</div>
            <div class="label">staypoints détectés</div>
        </div>

        <div class="card">
            <div class="value">{len(merged_staypoints)}</div>
            <div class="label">lieux uniques</div>
        </div>

        <div class="card">
            <div class="value">{total_trimmed_points}</div>
            <div class="label">après trimming</div>
        </div>

        <div class="card">
            <div class="value">{total_sampled_points}</div>
            <div class="label">après sampling</div>
        </div>
    </div>

    <p>
        Staypoint : <b>{STAYPOINT_TIME_MIN} min</b> dans
        <b>{STAYPOINT_RADIUS_M} m</b><br>
        Regroupement lieux : <b>{MERGE_STAYPOINT_RADIUS_M} m</b><br>
        Trimming local : <b>{LOCAL_RADIUS_M} m</b><br>
        Sampling : <b>1 point sur {SAMPLE_STEP}</b>
    </p>

    <p>
        Rouge transparent = CSV brut complet<br>
        Couleurs = trajets séparés après traitement<br>
        Vert = départ brut · Rouge = arrivée brute<br>
        Bleu = premier point conservé · Noir = dernier point conservé<br>
        Jaune/noir = lieu unique regroupé
    </p>
</div>

{{% endmacro %}}
"""

    panel = MacroElement()
    panel._template = Template(panel_html)
    map_object.get_root().add_child(panel)


# =========================
# PIPELINE
# =========================

def get_user_id(csv_path):
    return os.path.splitext(os.path.basename(csv_path))[0]


def build_refined_dataframe(processed_trajectories, user_id, source_file):
    refined_frames = []

    for index, trajectory_data in enumerate(processed_trajectories):
        trajectory_id = index + 1
        trimmed = trajectory_data["trimmed"].copy()
        trimmed["USER_ID"] = user_id
        trimmed["SOURCE_FILE"] = source_file
        trimmed["TRAJECTORY_ID"] = trajectory_id
        trimmed["TRAJECTORY_UID"] = f"{user_id}_{trajectory_id}"
        trimmed["POINT_INDEX"] = range(len(trimmed))
        refined_frames.append(trimmed)

    if not refined_frames:
        return pd.DataFrame()

    return pd.concat(refined_frames, ignore_index=True)


def process_csv_file(csv_path, show_progress=False, verbose=False):
    start_time = time.time()
    user_id = get_user_id(csv_path)
    source_file = os.path.basename(csv_path)

    all_points = load_and_clean_data(csv_path)
    valid_points = len(all_points)

    before_presampling = len(all_points)
    all_points = light_presampling(
        all_points,
        max_distance_m=PRE_SAMPLE_MAX_DISTANCE_M,
        max_time_seconds=PRE_SAMPLE_MAX_TIME_SECONDS,
        show_progress=show_progress
    )
    after_presampling = len(all_points)

    trajectory_segments, staypoints = segment_trajectories_by_staypoints(
        all_points,
        show_progress=show_progress
    )
    merged_staypoints = merge_staypoints(staypoints, MERGE_STAYPOINT_RADIUS_M)

    processed_trajectories = []

    iterator = tqdm(
        trajectory_segments,
        desc="Traitement des trajets",
        unit="trajet",
        disable=not show_progress
    )

    for index, trajectory in enumerate(iterator):
        trimmed_trajectory = trim_start_end_zones(trajectory, LOCAL_RADIUS_M)

        if len(trimmed_trajectory) < MIN_POINTS_TRAJ:
            continue

        sampled_trajectory = (
            trimmed_trajectory
            .iloc[::SAMPLE_STEP]
            .copy()
            .reset_index(drop=True)
        )

        processed_trajectories.append({
            "raw": trajectory,
            "trimmed": trimmed_trajectory,
            "sampled": sampled_trajectory
        })

        if verbose:
            tqdm.write(
                f"   Trajet {index + 1:02d} | "
                f"{len(trajectory):4d} pts -> "
                f"{len(trimmed_trajectory):4d} pts -> "
                f"{len(sampled_trajectory):3d} pts"
            )

    refined_df = build_refined_dataframe(
        processed_trajectories,
        user_id=user_id,
        source_file=source_file
    )

    stats = {
        "user_id": user_id,
        "source_file": source_file,
        "valid_points": valid_points,
        "points_after_presampling": after_presampling,
        "presampling_removed_points": before_presampling - after_presampling,
        "trajectory_segments": len(trajectory_segments),
        "trajectories_kept": len(processed_trajectories),
        "staypoints": len(staypoints),
        "merged_staypoints": len(merged_staypoints),
        "refined_points": len(refined_df),
        "processing_seconds": time.time() - start_time,
        "error": ""
    }

    return {
        "all_points": all_points,
        "trajectory_segments": trajectory_segments,
        "staypoints": staypoints,
        "merged_staypoints": merged_staypoints,
        "processed_trajectories": processed_trajectories,
        "refined_df": refined_df,
        "stats": stats
    }


def process_csv_file_for_batch(csv_path):
    try:
        result = process_csv_file(csv_path, show_progress=False, verbose=False)
        return {
            "refined_df": result["refined_df"],
            "stats": result["stats"]
        }
    except Exception as exc:
        return {
            "refined_df": pd.DataFrame(),
            "stats": {
                "user_id": get_user_id(csv_path),
                "source_file": os.path.basename(csv_path),
                "valid_points": 0,
                "points_after_presampling": 0,
                "presampling_removed_points": 0,
                "trajectory_segments": 0,
                "trajectories_kept": 0,
                "staypoints": 0,
                "merged_staypoints": 0,
                "refined_points": 0,
                "processing_seconds": 0,
                "error": str(exc)
            }
        }


def process_csv_file_for_random_map(csv_path):
    try:
        result = process_csv_file(csv_path, show_progress=False, verbose=False)
        user_id = result["stats"]["user_id"]
        trajectories = []

        for index, trajectory_data in enumerate(result["processed_trajectories"]):
            sampled = trajectory_data["sampled"]

            if len(sampled) < 2:
                continue

            trajectory_id = index + 1
            points = sampled[["LATITUDE", "LONGITUDE"]].values.tolist()
            trajectories.append({
                "user_id": user_id,
                "trajectory_id": trajectory_id,
                "trajectory_uid": f"{user_id}_{trajectory_id}",
                "points": points,
                "raw_points": len(trajectory_data["raw"]),
                "trimmed_points": len(trajectory_data["trimmed"]),
                "sampled_points": len(sampled)
            })

        return {
            "trajectories": trajectories,
            "stats": result["stats"]
        }
    except Exception as exc:
        return {
            "trajectories": [],
            "stats": {
                "user_id": get_user_id(csv_path),
                "source_file": os.path.basename(csv_path),
                "valid_points": 0,
                "points_after_presampling": 0,
                "presampling_removed_points": 0,
                "trajectory_segments": 0,
                "trajectories_kept": 0,
                "staypoints": 0,
                "merged_staypoints": 0,
                "refined_points": 0,
                "processing_seconds": 0,
                "error": str(exc)
            }
        }


def run_single_pipeline(csv_name=CSV_NAME):
    pipeline_start_time = time.time()

    print("\n===== GPS PIPELINE =====\n")

    csv_path = os.path.join(DATA_DIR, csv_name)

    print(f"Fichier : {csv_path}")
    print(f"Existe  : {os.path.exists(csv_path)}\n")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    with tqdm(total=7, desc="Pipeline global", unit="étape") as pipeline_bar:
        print("\n[1/4] Traitement du CSV")
        result = process_csv_file(csv_path, show_progress=True, verbose=True)
        all_points = result["all_points"]
        trajectory_segments = result["trajectory_segments"]
        staypoints = result["staypoints"]
        merged_staypoints = result["merged_staypoints"]
        processed_trajectories = result["processed_trajectories"]
        refined_df = result["refined_df"]
        stats = result["stats"]

        if not processed_trajectories:
            raise ValueError("Aucun trajet exploitable après traitement.")

        print(f"      -> {stats['valid_points']} points valides")
        print(f"      -> {stats['points_after_presampling']} points après pré-sampling")
        print(f"      -> {stats['trajectory_segments']} trajets détectés")
        print(f"      -> {stats['trajectories_kept']} trajets conservés")
        print(f"      -> {stats['staypoints']} staypoints détectés")
        print(f"      -> {stats['merged_staypoints']} lieux uniques")
        pipeline_bar.update(4)

        print("\n[5/7] Construction de la carte")
        map_object = build_map(
            all_points,
            processed_trajectories,
            merged_staypoints,
            show_progress=True
        )

        inject_info_panel(
            map_object,
            all_points,
            trajectory_segments,
            processed_trajectories,
            staypoints,
            merged_staypoints
        )

        print("      -> Carte construite")
        pipeline_bar.update(1)

        print("\n[6/7] Sauvegarde du fichier HTML")
        output_path = os.path.abspath(OUTPUT_FILE)
        map_object.save(output_path)
        print(f"      -> {output_path}")
        pipeline_bar.update(1)

        print("\n[7/7] Export CSV des données raffinées")
        csv_output_name = OUTPUT_FILE.replace(".html", "_refined.csv")
        csv_output_path = os.path.abspath(csv_output_name)
        refined_df.to_csv(csv_output_path, index=False)

        print(f"      -> {len(refined_df)} points exportés")
        print(f"      -> {csv_output_path}")
        pipeline_bar.update(1)

    total_time = time.time() - pipeline_start_time

    print("\n===== RÉSUMÉ =====")
    print(f"Points valides       : {stats['valid_points']}")
    print(f"Trajets détectés     : {stats['trajectory_segments']}")
    print(f"Trajets conservés    : {stats['trajectories_kept']}")
    print(f"Staypoints détectés  : {stats['staypoints']}")
    print(f"Lieux uniques        : {stats['merged_staypoints']}")
    print(f"Fichier HTML généré  : {os.path.abspath(OUTPUT_FILE)}")
    print(f"Fichier CSV raffiné  : {csv_output_path}")
    print(f"Temps total          : {total_time:.2f} secondes")
    print("\nTerminé.")


def run_batch_pipeline(limit=None, workers=None):
    batch_start_time = time.time()
    csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    if limit is not None:
        csv_paths = csv_paths[:limit]

    if not csv_paths:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {DATA_DIR}")

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    output_path = os.path.abspath(BATCH_OUTPUT_FILE)
    summary_path = os.path.abspath(BATCH_SUMMARY_FILE)

    print("\n===== GPS PIPELINE BATCH =====\n")
    print(f"Dossier source : {DATA_DIR}")
    print(f"Fichiers CSV   : {len(csv_paths)}")
    print(f"Workers        : {workers}")
    print(f"Sortie CSV     : {output_path}")
    print(f"Résumé         : {summary_path}\n")

    stats_rows = []
    wrote_header = False
    global_trajectory_offset = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_csv_file_for_batch, csv_path): csv_path
            for csv_path in csv_paths
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Traitement batch",
            unit="csv"
        ):
            result = future.result()
            refined_df = result["refined_df"]
            stats = result["stats"]

            if not refined_df.empty:
                trajectory_uids = refined_df["TRAJECTORY_UID"].drop_duplicates()
                uid_to_global_id = {
                    uid: global_trajectory_offset + index + 1
                    for index, uid in enumerate(trajectory_uids)
                }
                refined_df["GLOBAL_TRAJECTORY_ID"] = (
                    refined_df["TRAJECTORY_UID"].map(uid_to_global_id)
                )
                global_trajectory_offset += len(trajectory_uids)

                refined_df.to_csv(
                    output_path,
                    mode="w" if not wrote_header else "a",
                    header=not wrote_header,
                    index=False
                )
                wrote_header = True

            stats_rows.append(stats)

    summary_df = pd.DataFrame(stats_rows).sort_values("source_file")
    summary_df.to_csv(summary_path, index=False)

    total_time = time.time() - batch_start_time
    errors = summary_df["error"].astype(bool).sum()

    print("\n===== RÉSUMÉ BATCH =====")
    print(f"CSV traités              : {len(summary_df)}")
    print(f"Erreurs                  : {errors}")
    print(f"Trajets conservés        : {summary_df['trajectories_kept'].sum()}")
    print(f"Points raffinés exportés : {summary_df['refined_points'].sum()}")
    print(f"Fichier global           : {output_path}")
    print(f"Résumé par CSV           : {summary_path}")
    print(f"Temps total              : {total_time:.2f} secondes")
    print("\nTerminé.")


def build_random_sample_map(trajectories, stats_rows, output_path, sample_size, seed):
    all_latitudes = [
        point[0]
        for trajectory in trajectories
        for point in trajectory["points"]
    ]
    all_longitudes = [
        point[1]
        for trajectory in trajectories
        for point in trajectory["points"]
    ]

    if not all_latitudes or not all_longitudes:
        raise ValueError("Aucune trajectoire exploitable pour construire la carte.")

    map_object = folium.Map(
        location=[
            sum(all_latitudes) / len(all_latitudes),
            sum(all_longitudes) / len(all_longitudes)
        ],
        zoom_start=10,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer("CartoDB positron", name="Fond clair").add_to(map_object)
    folium.TileLayer("CartoDB dark_matter", name="Fond sombre").add_to(map_object)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(map_object)

    for index, trajectory in enumerate(
        tqdm(trajectories, desc="Ajout des trajets à la carte", unit="trajet")
    ):
        color = COLORS[index % len(COLORS)]
        tooltip = (
            f"{trajectory['trajectory_uid']} | "
            f"{trajectory['raw_points']} pts bruts -> "
            f"{trajectory['trimmed_points']} trim -> "
            f"{trajectory['sampled_points']} affichés"
        )

        folium.PolyLine(
            trajectory["points"],
            color=color,
            weight=3,
            opacity=0.65,
            tooltip=tooltip
        ).add_to(map_object)

    Fullscreen().add_to(map_object)
    MiniMap(toggle_display=True, position="bottomright").add_to(map_object)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordonnées",
        num_digits=5
    ).add_to(map_object)
    folium.LayerControl(collapsed=True).add_to(map_object)

    map_object.fit_bounds([
        [min(all_latitudes), min(all_longitudes)],
        [max(all_latitudes), max(all_longitudes)]
    ])

    successful_users = sum(not row["error"] for row in stats_rows)
    panel_html = f"""
{{% macro html(this, kwargs) %}}
<style>
#gps-panel {{
    position: fixed;
    top: 20px;
    left: 20px;
    width: 320px;
    z-index: 9999;
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    padding: 16px;
    font-family: Arial, sans-serif;
    color: #0f172a;
}}
#gps-panel h2 {{
    margin: 0 0 10px;
    font-size: 17px;
}}
#gps-panel p {{
    margin: 6px 0;
    color: #475569;
    font-size: 13px;
    line-height: 1.45;
}}
</style>
<div id="gps-panel">
    <h2>Échantillon GPS aléatoire</h2>
    <p><b>{sample_size}</b> CSV tirés au hasard dans <b>gps_dataset</b>.</p>
    <p><b>{successful_users}</b> utilisateurs traités sans erreur.</p>
    <p><b>{len(trajectories)}</b> trajets affichés après trimming et sampling.</p>
    <p>Seed : <b>{seed}</b></p>
</div>
{{% endmacro %}}
"""

    panel = MacroElement()
    panel._template = Template(panel_html)
    map_object.get_root().add_child(panel)
    map_object.save(output_path)


def run_random_map_pipeline(sample_size=50, seed=42, workers=None, output_file=None):
    start_time = time.time()
    csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    if not csv_paths:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {DATA_DIR}")

    sample_size = min(sample_size, len(csv_paths))
    rng = random.Random(seed)
    selected_paths = rng.sample(csv_paths, sample_size)

    if workers is None:
        workers = max(1, min(sample_size, (os.cpu_count() or 2) - 1))

    output_path = os.path.abspath(output_file or RANDOM_MAP_OUTPUT_FILE)
    summary_path = os.path.abspath(RANDOM_MAP_SUMMARY_FILE)

    print("\n===== CARTE GPS ALÉATOIRE =====\n")
    print(f"Dossier source : {DATA_DIR}")
    print(f"CSV tirés      : {sample_size}")
    print(f"Seed           : {seed}")
    print(f"Workers        : {workers}")
    print(f"Carte HTML     : {output_path}")
    print(f"Résumé         : {summary_path}\n")

    all_trajectories = []
    stats_rows = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_csv_file_for_random_map, csv_path): csv_path
            for csv_path in selected_paths
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Traitement carte random",
            unit="csv"
        ):
            result = future.result()
            all_trajectories.extend(result["trajectories"])
            stats_rows.append(result["stats"])

    summary_df = pd.DataFrame(stats_rows).sort_values("source_file")
    summary_df.to_csv(summary_path, index=False)

    build_random_sample_map(
        all_trajectories,
        stats_rows,
        output_path=output_path,
        sample_size=sample_size,
        seed=seed
    )

    errors = summary_df["error"].astype(bool).sum()
    total_time = time.time() - start_time

    print("\n===== RÉSUMÉ CARTE RANDOM =====")
    print(f"CSV sélectionnés       : {sample_size}")
    print(f"Erreurs                : {errors}")
    print(f"Trajets affichés       : {len(all_trajectories)}")
    print(f"Carte HTML             : {output_path}")
    print(f"Résumé par CSV         : {summary_path}")
    print(f"Temps total            : {total_time:.2f} secondes")
    print("\nTerminé.")


def load_random_trajectories_from_refined_csv(refined_csv_path, sample_size=50, seed=42):
    if not os.path.exists(refined_csv_path):
        raise FileNotFoundError(refined_csv_path)

    refined_df = pd.read_csv(refined_csv_path)

    if refined_df.empty:
        raise ValueError(f"Le fichier raffiné est vide : {refined_csv_path}")

    required_columns = {"USER_ID", "TRAJECTORY_UID", "LATITUDE", "LONGITUDE"}
    missing_columns = required_columns - set(refined_df.columns)

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le CSV raffiné : "
            + ", ".join(sorted(missing_columns))
        )

    user_ids = sorted(refined_df["USER_ID"].dropna().unique())
    sample_size = min(sample_size, len(user_ids))
    selected_users = set(random.Random(seed).sample(user_ids, sample_size))
    selected_df = refined_df[refined_df["USER_ID"].isin(selected_users)].copy()

    if "POINT_INDEX" in selected_df.columns:
        selected_df = selected_df.sort_values(["TRAJECTORY_UID", "POINT_INDEX"])
    elif "DATETIME" in selected_df.columns:
        selected_df = selected_df.sort_values(["TRAJECTORY_UID", "DATETIME"])

    trajectories = []

    for trajectory_uid, trajectory_df in selected_df.groupby("TRAJECTORY_UID", sort=False):
        sampled_df = trajectory_df.iloc[::SAMPLE_STEP]

        if len(sampled_df) < 2:
            continue

        user_id = sampled_df["USER_ID"].iloc[0]
        trajectory_id = (
            sampled_df["TRAJECTORY_ID"].iloc[0]
            if "TRAJECTORY_ID" in sampled_df.columns
            else ""
        )

        trajectories.append({
            "user_id": user_id,
            "trajectory_id": trajectory_id,
            "trajectory_uid": trajectory_uid,
            "points": sampled_df[["LATITUDE", "LONGITUDE"]].values.tolist(),
            "raw_points": len(trajectory_df),
            "trimmed_points": len(trajectory_df),
            "sampled_points": len(sampled_df)
        })

    stats_rows = [
        {"error": "", "user_id": user_id}
        for user_id in selected_users
    ]

    return trajectories, stats_rows, sample_size


def run_random_map_from_batch_output(sample_size=DEFAULT_RANDOM_MAP_USERS, seed=42, output_file=None):
    start_time = time.time()
    refined_csv_path = os.path.abspath(BATCH_OUTPUT_FILE)
    output_path = os.path.abspath(output_file or RANDOM_MAP_OUTPUT_FILE)

    print("\n===== CARTE GPS DEPUIS LE BATCH =====\n")
    print(f"CSV raffiné    : {refined_csv_path}")
    print(f"Utilisateurs   : {sample_size}")
    print(f"Seed           : {seed}")
    print(f"Carte HTML     : {output_path}\n")

    trajectories, stats_rows, actual_sample_size = load_random_trajectories_from_refined_csv(
        refined_csv_path,
        sample_size=sample_size,
        seed=seed
    )

    build_random_sample_map(
        trajectories,
        stats_rows,
        output_path=output_path,
        sample_size=actual_sample_size,
        seed=seed
    )

    total_time = time.time() - start_time

    print("\n===== RÉSUMÉ CARTE BATCH =====")
    print(f"Utilisateurs tirés     : {actual_sample_size}")
    print(f"Trajets affichés       : {len(trajectories)}")
    print(f"Carte HTML             : {output_path}")
    print(f"Temps total            : {total_time:.2f} secondes")
    print("\nTerminé.")


def run_default_pipeline(
    workers=DEFAULT_WORKERS,
    sample_size=DEFAULT_RANDOM_MAP_USERS,
    seed=42,
    recompute_map=False
):
    if os.path.exists(BATCH_OUTPUT_FILE) and not recompute_map:
        run_random_map_from_batch_output(
            sample_size=sample_size,
            seed=seed,
            output_file=RANDOM_MAP_OUTPUT_FILE
        )
        return

    run_random_map_pipeline(
        sample_size=sample_size,
        seed=seed,
        workers=workers,
        output_file=RANDOM_MAP_OUTPUT_FILE
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de traitement des trajectoires GPS."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Traite plusieurs CSV de gps_dataset en parallèle."
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Traite un seul CSV et génère la carte détaillée."
    )
    parser.add_argument(
        "--random-map",
        type=int,
        default=None,
        metavar="N",
        help="Génère une carte avec N CSV tirés aléatoirement dans gps_dataset."
    )
    parser.add_argument(
        "--csv",
        default=CSV_NAME,
        help="Nom du CSV à traiter en mode fichier unique."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximum de CSV à traiter en mode batch."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Nombre de processus parallèles."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed utilisée pour le tirage aléatoire de --random-map."
    )
    parser.add_argument(
        "--random-map-output",
        default=None,
        help="Nom du fichier HTML à générer avec --random-map."
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Genere seulement la carte depuis all_refined_trajectories.csv existant."
    )
    parser.add_argument(
        "--recompute-map",
        action="store_true",
        help="Force --random-map a retraiter les CSV bruts au lieu de reutiliser le batch."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.map_only:
        run_random_map_from_batch_output(
            sample_size=args.random_map or DEFAULT_RANDOM_MAP_USERS,
            seed=args.seed,
            output_file=args.random_map_output
        )
    elif args.single:
        run_single_pipeline(csv_name=args.csv)
    elif args.random_map is not None:
        if os.path.exists(BATCH_OUTPUT_FILE) and not args.recompute_map:
            run_random_map_from_batch_output(
                sample_size=args.random_map,
                seed=args.seed,
                output_file=args.random_map_output
            )
        else:
            run_random_map_pipeline(
                sample_size=args.random_map,
                seed=args.seed,
                workers=args.workers,
                output_file=args.random_map_output
            )
    elif args.batch:
        run_batch_pipeline(limit=args.limit, workers=args.workers)
    else:
        run_default_pipeline(
            workers=args.workers,
            sample_size=DEFAULT_RANDOM_MAP_USERS,
            seed=args.seed,
            recompute_map=args.recompute_map
        )


if __name__ == "__main__":
    main()
