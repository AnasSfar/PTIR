import os
import math
import pandas as pd
import folium
from folium.plugins import Fullscreen, MiniMap, MousePosition, MeasureControl
from branca.element import Element

DATA_DIR = r"C:\Users\sfara\Documents\GitHub\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
CSV_NAME = "12_3332.csv"

OUTPUT_HTML = "traj_12_3332_trim_comparison.html"

LOCAL_RADII = [50, 80, 120, 150, 200]  # mètres
TRIM_TIMES = [0, 1, 2, 3, 5]           # minutes

IDF_BOUNDS = dict(min_lat=48.10, max_lat=49.25, min_lon=1.40, max_lon=3.60)


def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_csv(path):
    df = pd.read_csv(path)

    if not {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        raise ValueError("Colonnes LATITUDE / LONGITUDE manquantes.")

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
        df = df.dropna(subset=["DATETIME"])
        df = df.sort_values("DATETIME")

    df = df.reset_index(drop=True)

    print(f"Points valides chargés : {len(df)}")
    return df


def compute_metrics(df):
    df = df.copy().reset_index(drop=True)
    n = len(df)

    dist_m = [0.0] * n

    for i in range(n - 1):
        dist_m[i] = haversine(
            df.at[i, "LATITUDE"],
            df.at[i, "LONGITUDE"],
            df.at[i + 1, "LATITUDE"],
            df.at[i + 1, "LONGITUDE"]
        )

    df["dist_m"] = dist_m

    lat_start, lon_start = df.at[0, "LATITUDE"], df.at[0, "LONGITUDE"]
    lat_end, lon_end = df.at[n - 1, "LATITUDE"], df.at[n - 1, "LONGITUDE"]

    df["dist_start"] = df.apply(
        lambda row: haversine(
            lat_start,
            lon_start,
            row["LATITUDE"],
            row["LONGITUDE"]
        ),
        axis=1
    )

    df["dist_end"] = df.apply(
        lambda row: haversine(
            lat_end,
            lon_end,
            row["LATITUDE"],
            row["LONGITUDE"]
        ),
        axis=1
    )

    return df


def trim_local_zone(df, local_radius_m, trim_minutes):
    """
    Nouveau trimming :
    - supprime TOUS les points dans la zone locale de départ
    - supprime TOUS les points dans la zone locale d'arrivée
    - supprime aussi les premières / dernières minutes si trim_minutes > 0

    Contrairement à l’ancien trim, on ne coupe pas seulement les extrémités.
    Ici, on enlève vraiment les points proches du départ ou de l’arrivée,
    même s’ils apparaissent au milieu de la trajectoire.
    """

    df = compute_metrics(df)
    df = df.sort_values("DATETIME").reset_index(drop=True)

    start_time = df["DATETIME"].iloc[0]
    end_time = df["DATETIME"].iloc[-1]

    start_limit = start_time + pd.Timedelta(minutes=trim_minutes)
    end_limit = end_time - pd.Timedelta(minutes=trim_minutes)

    mask_start_zone = df["dist_start"] <= local_radius_m
    mask_end_zone = df["dist_end"] <= local_radius_m

    if trim_minutes > 0:
        mask_start_time = df["DATETIME"] <= start_limit
        mask_end_time = df["DATETIME"] >= end_limit
    else:
        mask_start_time = pd.Series(False, index=df.index)
        mask_end_time = pd.Series(False, index=df.index)

    remove_mask = (
        mask_start_zone |
        mask_end_zone |
        mask_start_time |
        mask_end_time
    )

    df_trimmed = df[~remove_mask].copy().reset_index(drop=True)

    removed_start = int((mask_start_zone | mask_start_time).sum())
    removed_end = int((mask_end_zone | mask_end_time).sum())
    removed_total = int(remove_mask.sum())

    return df_trimmed, removed_start, removed_end, removed_total


def split_segments(df, max_gap_seconds=60, max_gap_meters=200):
    """
    Après suppression des points, on évite de relier artificiellement
    deux morceaux éloignés par une grande ligne droite.
    """

    if len(df) < 2:
        return []

    df = df.sort_values("DATETIME").reset_index(drop=True)

    segments = []
    current = []

    for i, row in df.iterrows():
        point = [row["LATITUDE"], row["LONGITUDE"]]

        if i > 0:
            prev = df.iloc[i - 1]

            dt = (row["DATETIME"] - prev["DATETIME"]).total_seconds()
            dist = haversine(
                prev["LATITUDE"],
                prev["LONGITUDE"],
                row["LATITUDE"],
                row["LONGITUDE"]
            )

            if dt > max_gap_seconds or dist > max_gap_meters:
                if len(current) >= 2:
                    segments.append(current)
                current = []

        current.append(point)

    if len(current) >= 2:
        segments.append(current)

    return segments


def make_single_map_html(df_raw, df_trim, r, t, removed_start, removed_end, removed_total):
    n_before = len(df_raw)
    n_after = len(df_trim)
    pct = 100 * removed_total / n_before if n_before else 0

    pts_raw = df_raw[["LATITUDE", "LONGITUDE"]].values.tolist()
    segments_trim = split_segments(df_trim)

    center = [df_raw["LATITUDE"].mean(), df_raw["LONGITUDE"].mean()]

    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        width="100%",
        height="420px"
    )

    folium.PolyLine(
        pts_raw,
        color="#ef4444",
        weight=3,
        opacity=0.35,
        tooltip="Trajectoire brute"
    ).add_to(m)

    for seg in segments_trim:
        folium.PolyLine(
            seg,
            color="#22c55e",
            weight=3,
            opacity=0.95,
            tooltip=f"Après trimming r={r}m / t={t}min"
        ).add_to(m)

    folium.Circle(
        location=pts_raw[0],
        radius=r,
        color="#3b82f6",
        weight=1,
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=0.08,
        tooltip=f"Zone départ {r}m"
    ).add_to(m)

    folium.Circle(
        location=pts_raw[-1],
        radius=r,
        color="#3b82f6",
        weight=1,
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=0.08,
        tooltip=f"Zone arrivée {r}m"
    ).add_to(m)

    folium.CircleMarker(
        location=pts_raw[0],
        radius=5,
        color="#dc2626",
        fill=True,
        fill_color="#fca5a5",
        fill_opacity=0.8,
        tooltip="Départ brut"
    ).add_to(m)

    folium.CircleMarker(
        location=pts_raw[-1],
        radius=5,
        color="#dc2626",
        fill=True,
        fill_color="#fca5a5",
        fill_opacity=0.8,
        tooltip="Arrivée brute"
    ).add_to(m)

    if len(df_trim) > 0:
        pts_trim = df_trim[["LATITUDE", "LONGITUDE"]].values.tolist()

        folium.CircleMarker(
            location=pts_trim[0],
            radius=6,
            color="#1d4ed8",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=1,
            tooltip="Premier point conservé"
        ).add_to(m)

        folium.CircleMarker(
            location=pts_trim[-1],
            radius=6,
            color="#111827",
            fill=True,
            fill_color="#374151",
            fill_opacity=1,
            tooltip="Dernier point conservé"
        ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    m.fit_bounds(
        [
            [df_raw["LATITUDE"].min(), df_raw["LONGITUDE"].min()],
            [df_raw["LATITUDE"].max(), df_raw["LONGITUDE"].max()]
        ],
        max_zoom=15
    )

    map_html = m.get_root().render()

    card = f"""
    <section class="map-card">
        <div class="map-header">
            <h2>r = {r} m · t = {t} min</h2>
            <p>
                Avant : <b>{n_before}</b> pts ·
                Après : <b>{n_after}</b> pts ·
                Supprimés : <b>{removed_total}</b> ({pct:.2f}%)
                <br>
                Zone départ / temps début : <b>{removed_start}</b> pts ·
                Zone arrivée / temps fin : <b>{removed_end}</b> pts ·
                Segments verts : <b>{len(segments_trim)}</b>
            </p>
        </div>
        <div class="map-box">
            {map_html}
        </div>
    </section>
    """

    return card, {
        "radius_m": r,
        "trim_minutes": t,
        "points_before": n_before,
        "points_after": n_after,
        "removed_points": removed_total,
        "removed_pct": pct,
        "removed_start_or_time": removed_start,
        "removed_end_or_time": removed_end,
        "green_segments": len(segments_trim)
    }


def main():
    csv_path = os.path.join(DATA_DIR, CSV_NAME)

    print("Fichier :", csv_path)
    print("Existe :", os.path.exists(csv_path))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df_raw = load_csv(csv_path)

    if len(df_raw) < 10:
        raise ValueError("Trop peu de points.")

    cards = []
    rows = []

    for r in LOCAL_RADII:
        for t in TRIM_TIMES:
            print(f"Test r={r}m / t={t}min")

            df_trim, removed_start, removed_end, removed_total = trim_local_zone(
                df_raw,
                local_radius_m=r,
                trim_minutes=t
            )

            card_html, row = make_single_map_html(
                df_raw=df_raw,
                df_trim=df_trim,
                r=r,
                t=t,
                removed_start=removed_start,
                removed_end=removed_end,
                removed_total=removed_total
            )

            cards.append(card_html)
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("trim_comparison_summary.csv", index=False, encoding="utf-8")

    summary_table = summary_df.to_html(index=False, classes="summary-table", float_format="%.2f")

    html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Comparaison trimming — {CSV_NAME}</title>

    <style>
        body {{
            margin: 0;
            background: #f8fafc;
            color: #0f172a;
            font-family: Arial, sans-serif;
        }}

        header {{
            padding: 28px 36px;
            background: white;
            border-bottom: 1px solid #e2e8f0;
            position: sticky;
            top: 0;
            z-index: 9999;
        }}

        header h1 {{
            margin: 0 0 8px;
            font-size: 24px;
        }}

        header p {{
            margin: 0;
            color: #475569;
            line-height: 1.5;
        }}

        .legend {{
            margin-top: 12px;
            display: flex;
            gap: 18px;
            font-size: 14px;
        }}

        .red {{ color: #ef4444; font-weight: bold; }}
        .green {{ color: #22c55e; font-weight: bold; }}
        .blue {{ color: #3b82f6; font-weight: bold; }}

        .summary {{
            padding: 24px 36px;
        }}

        .summary-table {{
            border-collapse: collapse;
            background: white;
            width: 100%;
            font-size: 13px;
        }}

        .summary-table th,
        .summary-table td {{
            border: 1px solid #e2e8f0;
            padding: 8px 10px;
            text-align: center;
        }}

        .summary-table th {{
            background: #f1f5f9;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 24px;
            padding: 0 36px 36px;
        }}

        .map-card {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
        }}

        .map-header {{
            padding: 14px 16px;
            border-bottom: 1px solid #e2e8f0;
        }}

        .map-header h2 {{
            margin: 0 0 6px;
            font-size: 17px;
        }}

        .map-header p {{
            margin: 0;
            color: #475569;
            font-size: 13px;
            line-height: 1.5;
        }}

        .map-box {{
            height: 420px;
        }}

        .map-box > div {{
            height: 420px !important;
        }}
    </style>
</head>

<body>
    <header>
        <h1>Comparaison du trimming local — {CSV_NAME}</h1>
        <p>
            Chaque bloc teste une combinaison : rayon spatial <b>r</b> + trim temporel <b>t</b>.
            Cette version supprime réellement tous les points présents dans les zones locales,
            puis sépare les morceaux restants en segments pour éviter les fausses lignes droites.
        </p>

        <div class="legend">
            <span class="red">Rouge = brut</span>
            <span class="green">Vert = après suppression locale</span>
            <span class="blue">Bleu = zone locale</span>
        </div>
    </header>

    <div class="summary">
        <h2>Résumé des tests</h2>
        {summary_table}
    </div>

    <main class="grid">
        {''.join(cards)}
    </main>
</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("\nHTML unique créé :", os.path.abspath(OUTPUT_HTML))
    print("Résumé CSV créé :", os.path.abspath("trim_comparison_summary.csv"))


if __name__ == "__main__":
    main()