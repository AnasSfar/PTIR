import os
import math
import pandas as pd
import folium
from branca.element import Element

DATA_DIR = r"C:\Users\sfara\Documents\GitHub\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
CSV_NAME = "12_3332.csv"

OUTPUT_HTML = "traj_12_3332_sampling_comparison.html"

SAMPLING_STEPS = [1, 5, 10, 20, 50, 100]

IDF_BOUNDS = dict(min_lat=48.10, max_lat=49.25, min_lon=1.40, max_lon=3.60)


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

    return df.reset_index(drop=True)


def make_map_block(df_raw, step):
    df_sampled = df_raw.iloc[::step].copy().reset_index(drop=True)

    n_before = len(df_raw)
    n_after = len(df_sampled)
    pct_kept = 100 * n_after / n_before if n_before else 0

    pts_raw = df_raw[["LATITUDE", "LONGITUDE"]].values.tolist()
    pts_sampled = df_sampled[["LATITUDE", "LONGITUDE"]].values.tolist()

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

    # Trajectoire brute en rouge clair
    folium.PolyLine(
        pts_raw,
        color="#ef4444",
        weight=2,
        opacity=0.25,
        tooltip="Trajectoire brute"
    ).add_to(m)

    # Trajectoire samplée en vert
    folium.PolyLine(
        pts_sampled,
        color="#22c55e",
        weight=3,
        opacity=0.95,
        tooltip=f"Sampling step = {step}"
    ).add_to(m)

    # Points samplés
    for lat, lon in pts_sampled:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.7,
            opacity=0.7
        ).add_to(m)

    # Départ / arrivée
    folium.CircleMarker(
        location=pts_sampled[0],
        radius=6,
        color="#1d4ed8",
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=1,
        tooltip="Départ"
    ).add_to(m)

    folium.CircleMarker(
        location=pts_sampled[-1],
        radius=6,
        color="#111827",
        fill=True,
        fill_color="#374151",
        fill_opacity=1,
        tooltip="Arrivée"
    ).add_to(m)

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
            <h2>Sampling step = {step}</h2>
            <p>
                Points avant : <b>{n_before}</b> ·
                Points après : <b>{n_after}</b> ·
                Conservés : <b>{pct_kept:.2f}%</b>
            </p>
        </div>
        <div class="map-box">
            {map_html}
        </div>
    </section>
    """

    return card, {
        "sampling_step": step,
        "points_before": n_before,
        "points_after": n_after,
        "kept_pct": pct_kept
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

    for step in SAMPLING_STEPS:
        print(f"Test sampling step = {step}")

        card_html, row = make_map_block(df_raw, step)
        cards.append(card_html)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("sampling_comparison_summary.csv", index=False, encoding="utf-8")

    summary_table = summary_df.to_html(
        index=False,
        classes="summary-table",
        float_format="%.2f"
    )

    html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Comparaison sampling — {CSV_NAME}</title>

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
        <h1>Comparaison du sampling — {CSV_NAME}</h1>
        <p>
            Chaque bloc teste un niveau de sous-échantillonnage.
            L’objectif est de réduire le nombre de points tout en conservant la forme générale de la trajectoire.
        </p>

        <div class="legend">
            <span class="red">Rouge = trajectoire brute</span>
            <span class="green">Vert = trajectoire samplée</span>
            <span class="blue">Bleu = points conservés</span>
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

    print("\nHTML créé :", os.path.abspath(OUTPUT_HTML))
    print("Résumé CSV créé :", os.path.abspath("sampling_comparison_summary.csv"))


if __name__ == "__main__":
    main()