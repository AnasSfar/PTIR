import os
import math
import time
import pandas as pd
import folium
from tqdm import tqdm
from folium.plugins import Fullscreen, MiniMap, MousePosition, MeasureControl
from branca.element import Template, MacroElement

# =========================
# CONFIG
# =========================

DATA_DIR = r""
CSV_NAME = "12_3332.csv"

OUTPUT_FILE = "map_12_3332_segmented_pipeline.html"

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


def light_presampling(data, max_distance_m=10, max_time_seconds=10):
    """
    Pré-sampling léger :
    fusionne les points consécutifs très proches dans l'espace et dans le temps.
    """

    if len(data) < 2:
        return data.copy()

    reduced_points = []
    buffer_points = [data.iloc[0]]

    for i in tqdm(range(1, len(data)), desc="Pré-sampling léger", unit="pt"):
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

def segment_trajectories_by_staypoints(data):
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
        unit="pt"
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

def build_map(all_points, processed_trajectories, merged_staypoints):
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
        tqdm(processed_trajectories, desc="Ajout des trajets à la carte", unit="trajet")
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
# MAIN
# =========================

def main():
    pipeline_start_time = time.time()

    print("\n===== GPS PIPELINE =====\n")

    csv_path = os.path.join(DATA_DIR, CSV_NAME)

    print(f"Fichier : {csv_path}")
    print(f"Existe  : {os.path.exists(csv_path)}\n")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    with tqdm(total=7, desc="Pipeline global", unit="étape") as pipeline_bar:
        print("\n[1/7] Chargement et nettoyage des données")
        all_points = load_and_clean_data(csv_path)
        print(f"      → {len(all_points)} points valides")
        pipeline_bar.update(1)

        print("\n[2/7] Pré-sampling léger")
        before_count = len(all_points)

        all_points = light_presampling(
            all_points,
            max_distance_m=PRE_SAMPLE_MAX_DISTANCE_M,
            max_time_seconds=PRE_SAMPLE_MAX_TIME_SECONDS
        )

        after_count = len(all_points)
        reduction_pct = 100 * (before_count - after_count) / before_count

        print(f"      → {before_count} points → {after_count} points")
        print(f"      → réduction : {reduction_pct:.2f}%")
        pipeline_bar.update(1)

        print("\n[3/7] Détection des staypoints et segmentation")
        trajectory_segments, staypoints = segment_trajectories_by_staypoints(all_points)
        merged_staypoints = merge_staypoints(staypoints, MERGE_STAYPOINT_RADIUS_M)

        print(f"      → {len(trajectory_segments)} trajets détectés")
        print(f"      → {len(staypoints)} staypoints détectés")
        print(f"      → {len(merged_staypoints)} lieux uniques après regroupement")
        pipeline_bar.update(1)

        print("\n[4/7] Trimming + sampling des trajets")
        processed_trajectories = []

        for index, trajectory in enumerate(
            tqdm(trajectory_segments, desc="Traitement des trajets", unit="trajet")
        ):
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

            tqdm.write(
                f"   Trajet {index + 1:02d} | "
                f"{len(trajectory):4d} pts → "
                f"{len(trimmed_trajectory):4d} pts → "
                f"{len(sampled_trajectory):3d} pts"
            )

        if not processed_trajectories:
            raise ValueError("Aucun trajet exploitable après traitement.")

        print(f"      → {len(processed_trajectories)} trajets conservés")
        pipeline_bar.update(1)

        print("\n[5/7] Construction de la carte")
        map_object = build_map(
            all_points,
            processed_trajectories,
            merged_staypoints
        )

        inject_info_panel(
            map_object,
            all_points,
            trajectory_segments,
            processed_trajectories,
            staypoints,
            merged_staypoints
        )

        print("      → Carte construite")
        pipeline_bar.update(1)

        print("\n[6/7] Sauvegarde du fichier HTML")
        output_path = os.path.abspath(OUTPUT_FILE)
        map_object.save(output_path)
        print(f"      → {output_path}")
        pipeline_bar.update(1)

        print("\n[7/7] Export CSV des données raffinées")
        refined_frames = []
        for index, trajectory_data in enumerate(processed_trajectories):
            trimmed = trajectory_data["trimmed"].copy()
            trimmed["TRAJECTORY_ID"] = index + 1
            refined_frames.append(trimmed)

        refined_df = pd.concat(refined_frames, ignore_index=True)

        csv_output_name = OUTPUT_FILE.replace(".html", "_refined.csv")
        csv_output_path = os.path.abspath(csv_output_name)
        refined_df.to_csv(csv_output_path, index=False)

        print(f"      → {len(refined_df)} points exportés")
        print(f"      → {csv_output_path}")
        pipeline_bar.update(1)

    total_time = time.time() - pipeline_start_time

    print("\n===== RÉSUMÉ =====")
    print(f"Points valides       : {len(all_points)}")
    print(f"Trajets détectés     : {len(trajectory_segments)}")
    print(f"Trajets conservés    : {len(processed_trajectories)}")
    print(f"Staypoints détectés  : {len(staypoints)}")
    print(f"Lieux uniques        : {len(merged_staypoints)}")
    print(f"Fichier HTML généré  : {os.path.abspath(OUTPUT_FILE)}")
    print(f"Fichier CSV raffiné  : {csv_output_path}")
    print(f"Temps total          : {total_time:.2f} secondes")
    print("\nTerminé.")


if __name__ == "__main__":
    main()