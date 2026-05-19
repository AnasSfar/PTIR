import argparse
import math
import os

import folium
import pandas as pd
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen, MiniMap, MousePosition
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "all_refined_trajectories.csv")
OUTPUT_FILE = "shared_trajectory_segments.csv"
MAP_FILE = "map_shared_trajectory_segments.html"

DEFAULT_CELL_SIZE_M = 100
DEFAULT_MIN_USERS = 2
DEFAULT_MIN_TRAJECTORIES = 2
DEFAULT_MAP_MAX_SEGMENTS = 3000

IDF_REFERENCE_LAT = 48.8566
METERS_PER_DEGREE_LAT = 111_320
METERS_PER_DEGREE_LON = 111_320 * math.cos(math.radians(IDF_REFERENCE_LAT))


def cell_indexes_from_lat_lon(lat, lon, cell_size_m):
    lat_index = math.floor((lat * METERS_PER_DEGREE_LAT) / cell_size_m)
    lon_index = math.floor((lon * METERS_PER_DEGREE_LON) / cell_size_m)
    return lat_index, lon_index


def cell_id(lat_index, lon_index):
    return f"{lat_index}:{lon_index}"


def cell_center(cell, cell_size_m):
    lat_index, lon_index = [int(value) for value in cell.split(":")]
    lat = ((lat_index + 0.5) * cell_size_m) / METERS_PER_DEGREE_LAT
    lon = ((lon_index + 0.5) * cell_size_m) / METERS_PER_DEGREE_LON
    return lat, lon


def load_points(input_path, max_trajectories=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    data = pd.read_csv(input_path)

    required_columns = {"USER_ID", "TRAJECTORY_UID", "LATITUDE", "LONGITUDE"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes : " + ", ".join(sorted(missing_columns))
        )

    if max_trajectories is not None:
        selected_uids = data["TRAJECTORY_UID"].drop_duplicates().head(max_trajectories)
        data = data[data["TRAJECTORY_UID"].isin(selected_uids)].copy()

    sort_columns = ["TRAJECTORY_UID"]
    if "POINT_INDEX" in data.columns:
        sort_columns.append("POINT_INDEX")
    elif "DATETIME" in data.columns:
        sort_columns.append("DATETIME")

    return data.sort_values(sort_columns).reset_index(drop=True)


def build_shared_segments(points_df, cell_size_m):
    segment_rows = []

    grouped = points_df.groupby("TRAJECTORY_UID", sort=False)
    for trajectory_uid, trajectory_df in tqdm(
        grouped,
        total=points_df["TRAJECTORY_UID"].nunique(),
        desc="Construction des segments communs",
        unit="trajet"
    ):
        user_id = trajectory_df["USER_ID"].iloc[0]

        transitions = []
        current_cell = None
        current_start_lat = None
        current_start_lon = None
        current_last_lat = None
        current_last_lon = None

        for row in trajectory_df.itertuples(index=False):
            lat_index, lon_index = cell_indexes_from_lat_lon(
                row.LATITUDE,
                row.LONGITUDE,
                cell_size_m
            )
            next_cell = cell_id(lat_index, lon_index)

            if current_cell is None:
                current_cell = next_cell
                current_start_lat = row.LATITUDE
                current_start_lon = row.LONGITUDE
                current_last_lat = row.LATITUDE
                current_last_lon = row.LONGITUDE
                continue

            if next_cell == current_cell:
                current_last_lat = row.LATITUDE
                current_last_lon = row.LONGITUDE
                continue

            transitions.append({
                "from_cell": current_cell,
                "to_cell": next_cell,
                "from_latitude": current_last_lat,
                "from_longitude": current_last_lon,
                "to_latitude": row.LATITUDE,
                "to_longitude": row.LONGITUDE,
            })

            current_cell = next_cell
            current_start_lat = row.LATITUDE
            current_start_lon = row.LONGITUDE
            current_last_lat = row.LATITUDE
            current_last_lon = row.LONGITUDE

        if not transitions:
            continue

        # Same segment repeated inside one trajectory counts once for that trajectory.
        trajectory_segments = {}
        for transition in transitions:
            from_cell = transition["from_cell"]
            to_cell = transition["to_cell"]

            if from_cell == to_cell:
                continue

            segment_key = (from_cell, to_cell)
            if segment_key not in trajectory_segments:
                trajectory_segments[segment_key] = transition

        for (from_cell, to_cell), transition in trajectory_segments.items():

            segment_rows.append({
                "SEGMENT_ID": f"{from_cell}->{to_cell}",
                "FROM_CELL": from_cell,
                "TO_CELL": to_cell,
                "FROM_LATITUDE": transition["from_latitude"],
                "FROM_LONGITUDE": transition["from_longitude"],
                "TO_LATITUDE": transition["to_latitude"],
                "TO_LONGITUDE": transition["to_longitude"],
                "USER_ID": user_id,
                "TRAJECTORY_UID": trajectory_uid,
            })

    if not segment_rows:
        return pd.DataFrame()

    segments = pd.DataFrame(segment_rows)

    aggregated = (
        segments
        .groupby([
            "SEGMENT_ID",
            "FROM_CELL",
            "TO_CELL",
        ])
        .agg(
            FROM_LATITUDE=("FROM_LATITUDE", "mean"),
            FROM_LONGITUDE=("FROM_LONGITUDE", "mean"),
            TO_LATITUDE=("TO_LATITUDE", "mean"),
            TO_LONGITUDE=("TO_LONGITUDE", "mean"),
            N_USERS=("USER_ID", "nunique"),
            N_TRAJECTORIES=("TRAJECTORY_UID", "nunique"),
            USERS=("USER_ID", lambda values: " ".join(sorted(set(map(str, values))))),
        )
        .reset_index()
        .sort_values(["N_USERS", "N_TRAJECTORIES"], ascending=False)
    )

    return aggregated


def filter_segments(segments_df, min_users, min_trajectories):
    return segments_df[
        (segments_df["N_USERS"] >= min_users)
        & (segments_df["N_TRAJECTORIES"] >= min_trajectories)
    ].copy()


def build_segment_map(segments_df, args):
    if segments_df.empty:
        print("Carte ignoree : aucun segment commun a afficher.")
        return

    map_segments = (
        segments_df
        .sort_values(["N_USERS", "N_TRAJECTORIES"], ascending=False)
        .head(args.map_max_segments)
        .copy()
    )

    center = [
        pd.concat([map_segments["FROM_LATITUDE"], map_segments["TO_LATITUDE"]]).mean(),
        pd.concat([map_segments["FROM_LONGITUDE"], map_segments["TO_LONGITUDE"]]).mean(),
    ]

    map_object = folium.Map(
        location=center,
        zoom_start=11,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer("CartoDB positron", name="Fond clair").add_to(map_object)
    folium.TileLayer("CartoDB dark_matter", name="Fond sombre").add_to(map_object)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(map_object)

    max_users = max(1, int(map_segments["N_USERS"].max()))

    for row in tqdm(
        map_segments.itertuples(index=False),
        total=len(map_segments),
        desc="Ajout des segments communs",
        unit="segment"
    ):
        user_ratio = row.N_USERS / max_users
        weight = 1.5 + 7 * user_ratio
        opacity = 0.25 + 0.55 * user_ratio

        tooltip = (
            f"{row.FROM_CELL} -> {row.TO_CELL} | "
            f"{row.N_USERS} utilisateurs | "
            f"{row.N_TRAJECTORIES} trajets"
        )

        folium.PolyLine(
            [
                [row.FROM_LATITUDE, row.FROM_LONGITUDE],
                [row.TO_LATITUDE, row.TO_LONGITUDE],
            ],
            color="#2563eb",
            weight=weight,
            opacity=opacity,
            tooltip=tooltip,
        ).add_to(map_object)

    Fullscreen().add_to(map_object)
    MiniMap(toggle_display=True, position="bottomright").add_to(map_object)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordonnees",
        num_digits=5
    ).add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)

    map_object.fit_bounds([
        [
            min(map_segments["FROM_LATITUDE"].min(), map_segments["TO_LATITUDE"].min()),
            min(map_segments["FROM_LONGITUDE"].min(), map_segments["TO_LONGITUDE"].min()),
        ],
        [
            max(map_segments["FROM_LATITUDE"].max(), map_segments["TO_LATITUDE"].max()),
            max(map_segments["FROM_LONGITUDE"].max(), map_segments["TO_LONGITUDE"].max()),
        ],
    ])

    panel_html = f"""
{{% macro html(this, kwargs) %}}
<style>
#flow-panel {{
    position: fixed;
    top: 20px;
    left: 20px;
    width: 340px;
    z-index: 9999;
    background: white;
    border-radius: 10px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    padding: 16px;
    font-family: Arial, sans-serif;
    color: #0f172a;
}}
#flow-panel h2 {{
    margin: 0 0 10px;
    font-size: 17px;
}}
#flow-panel p {{
    margin: 6px 0;
    color: #475569;
    font-size: 13px;
    line-height: 1.45;
}}
</style>
<div id="flow-panel">
    <h2>Segments GPS communs</h2>
    <p><b>{len(map_segments)}</b> segments affiches.</p>
    <p>Grille : <b>{args.cell_size_m} m</b></p>
    <p>Minimum : <b>{args.min_users}</b> utilisateurs et <b>{args.min_trajectories}</b> trajets</p>
    <p>Epaisseur = nombre d'utilisateurs distincts</p>
</div>
{{% endmacro %}}
"""

    panel = MacroElement()
    panel._template = Template(panel_html)
    map_object.get_root().add_child(panel)

    output_path = os.path.abspath(args.map_output)
    map_object.save(output_path)
    print(f"Carte segments       : {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agrege les trajectoires GPS en segments communs ponderes."
    )
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--map-output", default=MAP_FILE)
    parser.add_argument("--cell-size-m", type=int, default=DEFAULT_CELL_SIZE_M)
    parser.add_argument("--min-users", type=int, default=DEFAULT_MIN_USERS)
    parser.add_argument("--min-trajectories", type=int, default=DEFAULT_MIN_TRAJECTORIES)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--map-max-segments", type=int, default=DEFAULT_MAP_MAX_SEGMENTS)
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Regenere seulement la carte depuis le CSV de segments existant."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force le recalcul des segments meme si le CSV de sortie existe deja."
    )
    parser.add_argument("--no-map", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n===== SHARED TRAJECTORY SEGMENTS =====\n")
    print(f"Entree trajectoires  : {os.path.abspath(args.input)}")
    print(f"Taille cellule       : {args.cell_size_m} m")
    print(f"Min utilisateurs     : {args.min_users}")
    print(f"Min trajets          : {args.min_trajectories}\n")

    output_path = os.path.abspath(args.output)
    can_reuse_output = os.path.exists(output_path)

    if args.map_only and not can_reuse_output:
        raise FileNotFoundError(output_path)

    if args.map_only or (can_reuse_output and not args.recompute and not args.no_map):
        print(f"CSV segments existant : {output_path}")
        print("Generation de la carte uniquement, sans recalcul des segments.")
        filtered_df = pd.read_csv(output_path)
        if not args.no_map:
            build_segment_map(filtered_df, args)
        print("\nTermine.")
        return

    points_df = load_points(args.input, max_trajectories=args.max_trajectories)
    print(f"Points charges       : {len(points_df)}")
    print(f"Trajets charges      : {points_df['TRAJECTORY_UID'].nunique()}\n")

    segments_df = build_shared_segments(points_df, args.cell_size_m)
    print(f"Segments distincts   : {len(segments_df)}")

    filtered_df = filter_segments(
        segments_df,
        min_users=args.min_users,
        min_trajectories=args.min_trajectories
    )
    print(f"Segments conserves   : {len(filtered_df)}")

    filtered_df.to_csv(output_path, index=False)
    print(f"CSV segments         : {output_path}")

    if not args.no_map:
        build_segment_map(filtered_df, args)

    print("\nTermine.")


if __name__ == "__main__":
    main()
