import os
import math
import pandas as pd
import folium
from folium.plugins import Fullscreen, MiniMap, MousePosition, MeasureControl
from branca.element import Template, MacroElement

# =========================
# CONFIG
# =========================

DATA_DIR = r"C:\Users\sfara\Documents\GitHub\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
CSV_NAME = "12_3332.csv"

OUTPUT_FILE = "map_12_3332_segmented_pipeline.html"

STAYPOINT_TIME_MIN = 10
STAYPOINT_RADIUS_M = 50

LOCAL_RADIUS_M = 100
SAMPLE_STEP = 5
MIN_POINTS_TRAJ = 10

IDF_BOUNDS = dict(min_lat=48.10, max_lat=49.25, min_lon=1.40, max_lon=3.60)

COLORS = [
    "#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316",
    "#14b8a6", "#ec4899", "#84cc16", "#6366f1", "#facc15"
]


# =========================
# OUTILS
# =========================

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_csv(path):
    df = pd.read_csv(path)

    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    df = df[
        (df["LATITUDE"].between(-90, 90)) &
        (df["LONGITUDE"].between(-180, 180))
    ]

    df = df[
        (df["LATITUDE"].between(IDF_BOUNDS["min_lat"], IDF_BOUNDS["max_lat"])) &
        (df["LONGITUDE"].between(IDF_BOUNDS["min_lon"], IDF_BOUNDS["max_lon"]))
    ]

    if "UTC_DATE" in df.columns and "UTC_TIME" in df.columns:
        df["DATETIME"] = pd.to_datetime(
            df["UTC_DATE"].astype(str) + " " + df["UTC_TIME"].astype(str),
            errors="coerce"
        )
    elif "LOCAL_DATE" in df.columns and "LOCAL_TIME" in df.columns:
        df["DATETIME"] = pd.to_datetime(
            df["LOCAL_DATE"].astype(str) + " " + df["LOCAL_TIME"].astype(str),
            errors="coerce"
        )
    else:
        raise ValueError("Colonnes de temps manquantes.")

    df = df.dropna(subset=["DATETIME"])
    df = df.sort_values("DATETIME").reset_index(drop=True)

    return df


# =========================
# SEGMENTATION PAR STAYPOINTS
# =========================

def split_by_staypoints(df):
    """
    Découpe le CSV en plusieurs trajets.

    Principe :
    si l'utilisateur reste dans une zone de STAYPOINT_RADIUS_M
    pendant au moins STAYPOINT_TIME_MIN minutes,
    alors on considère que le trajet est terminé.
    """

    trajectories = []
    current_points = []

    i = 0
    n = len(df)

    while i < n:
        current_points.append(df.iloc[i])

        j = i + 1
        stay_found = False

        while j < n:
            dist = haversine(
                df.iloc[i]["LATITUDE"],
                df.iloc[i]["LONGITUDE"],
                df.iloc[j]["LATITUDE"],
                df.iloc[j]["LONGITUDE"]
            )

            duration = (
                df.iloc[j]["DATETIME"] - df.iloc[i]["DATETIME"]
            ).total_seconds()

            if dist > STAYPOINT_RADIUS_M:
                break

            if duration >= STAYPOINT_TIME_MIN * 60:
                stay_found = True
                break

            j += 1

        if stay_found:
            traj_df = pd.DataFrame(current_points)

            if len(traj_df) >= MIN_POINTS_TRAJ:
                trajectories.append(traj_df.reset_index(drop=True))

            current_points = []
            i = j + 1
        else:
            i += 1

    if current_points:
        traj_df = pd.DataFrame(current_points)

        if len(traj_df) >= MIN_POINTS_TRAJ:
            trajectories.append(traj_df.reset_index(drop=True))

    return trajectories


# =========================
# TRIMMING
# =========================

def trim_local_zone(df, radius_m):
    if len(df) < 2:
        return df.copy()

    df = df.copy().reset_index(drop=True)

    start = df.iloc[0]
    end = df.iloc[-1]

    df["dist_start"] = df.apply(
        lambda row: haversine(
            start["LATITUDE"], start["LONGITUDE"],
            row["LATITUDE"], row["LONGITUDE"]
        ),
        axis=1
    )

    df["dist_end"] = df.apply(
        lambda row: haversine(
            end["LATITUDE"], end["LONGITUDE"],
            row["LATITUDE"], row["LONGITUDE"]
        ),
        axis=1
    )

    df = df[
        (df["dist_start"] > radius_m) &
        (df["dist_end"] > radius_m)
    ]

    return df.reset_index(drop=True)


# =========================
# MAP
# =========================

def build_map(df_raw, processed_trajectories):
    center = [
        df_raw["LATITUDE"].mean(),
        df_raw["LONGITUDE"].mean()
    ]

    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer("CartoDB positron", name="Fond clair").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Fond sombre").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    raw_points = df_raw[["LATITUDE", "LONGITUDE"]].values.tolist()

    folium.PolyLine(
        raw_points,
        color="#ef4444",
        weight=3,
        opacity=0.18,
        tooltip="CSV brut complet"
    ).add_to(m)

    for idx, traj in enumerate(processed_trajectories):
        color = COLORS[idx % len(COLORS)]

        raw_traj = traj["raw"]
        trimmed = traj["trimmed"]
        sampled = traj["sampled"]

        if len(trimmed) >= 2:
            folium.PolyLine(
                trimmed[["LATITUDE", "LONGITUDE"]].values.tolist(),
                color=color,
                weight=4,
                opacity=0.9,
                tooltip=f"Trajet {idx + 1} après trimming"
            ).add_to(m)

        # points samplés
        for _, row in sampled.iterrows():
            folium.CircleMarker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                tooltip=f"Point samplé trajet {idx + 1}"
            ).add_to(m)

        if len(raw_traj) >= 2:
            raw_start = raw_traj.iloc[0]
            raw_end = raw_traj.iloc[-1]

            # zone départ brute
            folium.Circle(
                location=[raw_start["LATITUDE"], raw_start["LONGITUDE"]],
                radius=LOCAL_RADIUS_M,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.07,
                tooltip=f"Zone départ trajet {idx + 1}"
            ).add_to(m)

            # zone arrivée brute
            folium.Circle(
                location=[raw_end["LATITUDE"], raw_end["LONGITUDE"]],
                radius=LOCAL_RADIUS_M,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.07,
                tooltip=f"Zone arrivée trajet {idx + 1}"
            ).add_to(m)

            # départ brut
            folium.CircleMarker(
                location=[raw_start["LATITUDE"], raw_start["LONGITUDE"]],
                radius=7,
                color="#16a34a",
                fill=True,
                fill_color="#22c55e",
                fill_opacity=1,
                tooltip=f"Départ brut trajet {idx + 1}"
            ).add_to(m)

            # arrivée brute
            folium.CircleMarker(
                location=[raw_end["LATITUDE"], raw_end["LONGITUDE"]],
                radius=7,
                color="#991b1b",
                fill=True,
                fill_color="#ef4444",
                fill_opacity=1,
                tooltip=f"Arrivée brute trajet {idx + 1}"
            ).add_to(m)

        if len(trimmed) >= 2:
            trimmed_start = trimmed.iloc[0]
            trimmed_end = trimmed.iloc[-1]

            # premier point conservé après trimming
            folium.CircleMarker(
                location=[trimmed_start["LATITUDE"], trimmed_start["LONGITUDE"]],
                radius=5,
                color="#1d4ed8",
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=1,
                tooltip=f"Premier point conservé trajet {idx + 1}"
            ).add_to(m)

            # dernier point conservé après trimming
            folium.CircleMarker(
                location=[trimmed_end["LATITUDE"], trimmed_end["LONGITUDE"]],
                radius=5,
                color="#111827",
                fill=True,
                fill_color="#374151",
                fill_opacity=1,
                tooltip=f"Dernier point conservé trajet {idx + 1}"
            ).add_to(m)

    Fullscreen().add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordonnées",
        num_digits=5
    ).add_to(m)
    MeasureControl().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.fit_bounds([
        [df_raw["LATITUDE"].min(), df_raw["LONGITUDE"].min()],
        [df_raw["LATITUDE"].max(), df_raw["LONGITUDE"].max()]
    ])

    return m

def inject_panel(m, df_raw, trajectories, processed_trajectories):
    total_trimmed = sum(len(t["trimmed"]) for t in processed_trajectories)
    total_sampled = sum(len(t["sampled"]) for t in processed_trajectories)

    panel_html = f"""
{{% macro html(this, kwargs) %}}

<style>
#gps-panel {{
    position: fixed;
    top: 20px;
    left: 20px;
    width: 330px;
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
        Pipeline : nettoyage → staypoints → segmentation →
        trimming local → sampling.
    </p>

    <div class="stat">
        <div class="card">
            <div class="value">{len(df_raw)}</div>
            <div class="label">points bruts</div>
        </div>

        <div class="card">
            <div class="value">{len(trajectories)}</div>
            <div class="label">trajets détectés</div>
        </div>

        <div class="card">
            <div class="value">{total_trimmed}</div>
            <div class="label">après trimming</div>
        </div>

        <div class="card">
            <div class="value">{total_sampled}</div>
            <div class="label">après sampling</div>
        </div>
    </div>

    <p>
        Staypoint : <b>{STAYPOINT_TIME_MIN} min</b> dans
        <b>{STAYPOINT_RADIUS_M} m</b><br>
        Trimming local : <b>{LOCAL_RADIUS_M} m</b><br>
        Sampling : <b>1 point sur {SAMPLE_STEP}</b>
    </p>

    <p>
        Rouge transparent = CSV brut complet<br>
        Couleurs = trajets séparés après traitement
    </p>
</div>

{{% endmacro %}}
"""

    panel = MacroElement()
    panel._template = Template(panel_html)
    m.get_root().add_child(panel)


# =========================
# MAIN
# =========================

def main():
    csv_path = os.path.join(DATA_DIR, CSV_NAME)

    print("Fichier :", csv_path)
    print("Existe :", os.path.exists(csv_path))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df_raw = load_csv(csv_path)

    print("Points bruts valides :", len(df_raw))

    trajectories = split_by_staypoints(df_raw)

    print("Trajets détectés :", len(trajectories))

    processed_trajectories = []

    for i, traj in enumerate(trajectories):
        trimmed = trim_local_zone(traj, LOCAL_RADIUS_M)

        if len(trimmed) < MIN_POINTS_TRAJ:
            continue

        sampled = trimmed.iloc[::SAMPLE_STEP].copy().reset_index(drop=True)

        processed_trajectories.append({
            "raw": traj,
            "trimmed": trimmed,
            "sampled": sampled
        })

        print(
            f"Trajet {i + 1} : "
            f"{len(traj)} pts bruts → "
            f"{len(trimmed)} après trimming → "
            f"{len(sampled)} après sampling"
        )

    if not processed_trajectories:
        raise ValueError("Aucun trajet exploitable après traitement.")

    m = build_map(df_raw, processed_trajectories)
    inject_panel(m, df_raw, trajectories, processed_trajectories)

    output_path = os.path.abspath(OUTPUT_FILE)
    m.save(output_path)

    print("\nCarte générée :", output_path)


if __name__ == "__main__":
    main()