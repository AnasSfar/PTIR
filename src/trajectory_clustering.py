import argparse
import math
import os
from collections import defaultdict

import pandas as pd
import folium
from tqdm import tqdm
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen, MiniMap, MousePosition


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFINED_TRAJECTORIES_FILE = os.path.join(BASE_DIR, "all_refined_trajectories.csv")
INDIVIDUALS_FILE = os.path.join(
    BASE_DIR,
    "NetMob25CleanedData",
    "NetMob25CleanedData",
    "individuals_dataset.csv"
)

GENERAL_CLUSTERS_FILE = "trajectory_clusters_general.csv"
GENERAL_SUMMARY_FILE = "cluster_summary_general.csv"
PROFILE_CLUSTERS_FILE = "trajectory_clusters_by_profile.csv"
PROFILE_SUMMARY_FILE = "cluster_summary_by_profile.csv"
TRAJECTORY_FEATURES_FILE = "trajectory_grid_features.csv"
CLUSTER_MAP_FILE = "map_trajectory_clusters.html"

DEFAULT_CELL_SIZE_M = 500
DEFAULT_JACCARD_THRESHOLD = 0.45
DEFAULT_MIN_SHARED_CELLS = 3
DEFAULT_PROFILE_CRITERIA = [
    "SEX",
    "AGE_GROUP",
    "NAVIGO_SUB",
    "DRIVING_LICENCE",
    "NB_CAR",
    "AREA_NAME",
]
COLORS = [
    "#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316",
    "#14b8a6", "#ec4899", "#84cc16", "#6366f1", "#facc15",
    "#0ea5e9", "#10b981", "#f43f5e", "#8b5cf6", "#fb923c",
]

IDF_REFERENCE_LAT = 48.8566
METERS_PER_DEGREE_LAT = 111_320
METERS_PER_DEGREE_LON = 111_320 * math.cos(math.radians(IDF_REFERENCE_LAT))


def cell_id_from_lat_lon(lat, lon, cell_size_m):
    lat_index = math.floor((lat * METERS_PER_DEGREE_LAT) / cell_size_m)
    lon_index = math.floor((lon * METERS_PER_DEGREE_LON) / cell_size_m)
    return f"{lat_index}:{lon_index}"


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


def load_refined_points(input_path, max_trajectories=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    data = pd.read_csv(input_path)

    required_columns = {"USER_ID", "TRAJECTORY_UID", "LATITUDE", "LONGITUDE"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le CSV raffiné : "
            + ", ".join(sorted(missing_columns))
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


def build_trajectory_features(points_df, cell_size_m):
    rows = []

    grouped = points_df.groupby("TRAJECTORY_UID", sort=False)

    for trajectory_uid, trajectory_df in tqdm(
        grouped,
        total=points_df["TRAJECTORY_UID"].nunique(),
        desc="Construction des features",
        unit="trajet"
    ):
        cells = [
            cell_id_from_lat_lon(row.LATITUDE, row.LONGITUDE, cell_size_m)
            for row in trajectory_df.itertuples(index=False)
        ]

        compressed_cells = []
        for cell in cells:
            if not compressed_cells or compressed_cells[-1] != cell:
                compressed_cells.append(cell)

        unique_cells = sorted(set(compressed_cells))

        if len(unique_cells) < 2:
            continue

        rows.append({
            "TRAJECTORY_UID": trajectory_uid,
            "USER_ID": trajectory_df["USER_ID"].iloc[0],
            "TRAJECTORY_ID": (
                trajectory_df["TRAJECTORY_ID"].iloc[0]
                if "TRAJECTORY_ID" in trajectory_df.columns
                else ""
            ),
            "GLOBAL_TRAJECTORY_ID": (
                trajectory_df["GLOBAL_TRAJECTORY_ID"].iloc[0]
                if "GLOBAL_TRAJECTORY_ID" in trajectory_df.columns
                else ""
            ),
            "START_CELL": compressed_cells[0],
            "END_CELL": compressed_cells[-1],
            "N_POINTS": len(trajectory_df),
            "N_CELLS": len(unique_cells),
            "CELL_SEQUENCE": " ".join(compressed_cells),
            "CELL_SET": " ".join(unique_cells),
        })

    return pd.DataFrame(rows)


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value):
        parent = self.parent[value]

        if parent != value:
            self.parent[value] = self.find(parent)

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


def jaccard_similarity(left_cells, right_cells):
    intersection_size = len(left_cells & right_cells)

    if intersection_size == 0:
        return 0.0, 0

    union_size = len(left_cells | right_cells)
    return intersection_size / union_size, intersection_size


def cluster_feature_frame(
    features_df,
    jaccard_threshold,
    min_shared_cells,
    od_required=True
):
    if features_df.empty:
        return pd.DataFrame()

    features = features_df.copy().reset_index(drop=True)
    features["CELL_SET_OBJ"] = features["CELL_SET"].apply(lambda value: set(str(value).split()))

    trajectory_ids = features["TRAJECTORY_UID"].tolist()
    union_find = UnionFind(trajectory_ids)

    buckets = defaultdict(list)

    for row in features.itertuples(index=False):
        key = (row.START_CELL, row.END_CELL) if od_required else "ALL"
        buckets[key].append(row.TRAJECTORY_UID)

    cell_sets = dict(zip(features["TRAJECTORY_UID"], features["CELL_SET_OBJ"]))

    for bucket_ids in tqdm(
        buckets.values(),
        total=len(buckets),
        desc="Comparaison des trajets",
        unit="bucket"
    ):
        if len(bucket_ids) < 2:
            continue

        for i in range(len(bucket_ids)):
            left_uid = bucket_ids[i]
            left_cells = cell_sets[left_uid]

            for j in range(i + 1, len(bucket_ids)):
                right_uid = bucket_ids[j]
                similarity, shared_cells = jaccard_similarity(left_cells, cell_sets[right_uid])

                if shared_cells >= min_shared_cells and similarity >= jaccard_threshold:
                    union_find.union(left_uid, right_uid)

    root_to_cluster_id = {}
    cluster_ids = []

    for trajectory_uid in trajectory_ids:
        root = union_find.find(trajectory_uid)

        if root not in root_to_cluster_id:
            root_to_cluster_id[root] = len(root_to_cluster_id)

        cluster_ids.append(root_to_cluster_id[root])

    features["CLUSTER_ID"] = cluster_ids
    features = features.drop(columns=["CELL_SET_OBJ"])

    return features


def summarize_clusters(clustered_df, group_columns=None):
    if clustered_df.empty:
        return pd.DataFrame()

    if group_columns is None:
        group_columns = []

    group_columns = list(group_columns) + ["CLUSTER_ID"]

    summary = (
        clustered_df
        .groupby(group_columns)
        .agg(
            N_TRAJECTORIES=("TRAJECTORY_UID", "nunique"),
            N_USERS=("USER_ID", "nunique"),
            AVG_POINTS=("N_POINTS", "mean"),
            AVG_CELLS=("N_CELLS", "mean"),
            START_CELL=("START_CELL", "first"),
            END_CELL=("END_CELL", "first"),
        )
        .reset_index()
        .sort_values(["N_TRAJECTORIES", "N_USERS"], ascending=False)
    )

    return summary


def load_individuals(individuals_path):
    if not os.path.exists(individuals_path):
        raise FileNotFoundError(individuals_path)

    individuals = pd.read_csv(individuals_path)

    if "ID" not in individuals.columns:
        raise ValueError("La colonne ID est absente de individuals_dataset.csv")

    if "AGE" in individuals.columns:
        individuals["AGE_GROUP"] = individuals["AGE"].apply(age_group)

    return individuals


def run_general_clustering(features_df, args):
    clustered = cluster_feature_frame(
        features_df,
        jaccard_threshold=args.jaccard_threshold,
        min_shared_cells=args.min_shared_cells,
        od_required=not args.no_od_filter
    )

    summary = summarize_clusters(clustered)

    clustered.to_csv(args.general_output, index=False)
    summary.to_csv(args.general_summary_output, index=False)

    return clustered, summary


def run_profile_clustering(features_df, individuals_df, args):
    merged = features_df.merge(
        individuals_df,
        left_on="USER_ID",
        right_on="ID",
        how="left"
    )

    all_clustered = []

    for criterion in args.criteria:
        if criterion not in merged.columns:
            print(f"Critère ignoré, colonne absente : {criterion}")
            continue

        for value, group_df in merged.groupby(criterion, dropna=False):
            if len(group_df) < args.profile_min_trajectories:
                continue

            clustered_group = cluster_feature_frame(
                group_df[features_df.columns],
                jaccard_threshold=args.jaccard_threshold,
                min_shared_cells=args.min_shared_cells,
                od_required=not args.no_od_filter
            )

            if clustered_group.empty:
                continue

            clustered_group["CRITERION"] = criterion
            clustered_group["CRITERION_VALUE"] = value
            clustered_group["PROFILE_CLUSTER_ID"] = (
                criterion + "=" + str(value) + "::" + clustered_group["CLUSTER_ID"].astype(str)
            )
            all_clustered.append(clustered_group)

    if not all_clustered:
        return pd.DataFrame(), pd.DataFrame()

    profile_clusters = pd.concat(all_clustered, ignore_index=True)
    profile_summary_source = profile_clusters.copy()
    profile_summary_source["CLUSTER_ID"] = profile_summary_source["PROFILE_CLUSTER_ID"]
    profile_summary = summarize_clusters(
        profile_summary_source,
        group_columns=["CRITERION", "CRITERION_VALUE"]
    )

    profile_clusters.to_csv(args.profile_output, index=False)
    profile_summary.to_csv(args.profile_summary_output, index=False)

    return profile_clusters, profile_summary


def build_cluster_map(points_df, clusters_df, summary_df, args):
    if clusters_df.empty:
        print("Carte ignorée : aucun cluster à afficher.")
        return

    eligible_summary = summary_df[
        (summary_df["N_TRAJECTORIES"] >= args.map_min_cluster_size)
        & (summary_df["N_USERS"] >= args.map_min_users)
    ].copy()

    if eligible_summary.empty:
        print(
            "Aucun cluster collectif trouve avec les seuils de carte ; "
            "affichage des meilleurs clusters disponibles."
        )
        eligible_summary = summary_df[
            summary_df["N_TRAJECTORIES"] >= args.map_min_cluster_size
        ].copy()

    if eligible_summary.empty:
        eligible_summary = summary_df.copy()

    if args.map_rank_by == "users":
        map_sort_columns = ["N_USERS", "N_TRAJECTORIES", "AVG_CELLS"]
    else:
        map_sort_columns = ["N_TRAJECTORIES", "N_USERS", "AVG_CELLS"]

    top_cluster_ids = (
        eligible_summary
        .sort_values(map_sort_columns, ascending=False)
        .head(args.map_top_clusters)["CLUSTER_ID"]
        .tolist()
    )

    selected_clusters = clusters_df[
        clusters_df["CLUSTER_ID"].isin(top_cluster_ids)
    ].copy()

    selected_uids = []

    for cluster_id in top_cluster_ids:
        cluster_rows = selected_clusters[selected_clusters["CLUSTER_ID"] == cluster_id].copy()
        cluster_rows = cluster_rows.sort_values(["USER_ID", "N_CELLS", "N_POINTS"])

        balanced_uids = []
        for _, user_group in cluster_rows.groupby("USER_ID", sort=False):
            balanced_uids.extend(
                user_group["TRAJECTORY_UID"]
                .head(args.map_max_trajectories_per_user)
                .tolist()
            )

        selected_uid_set = set(balanced_uids)
        if len(balanced_uids) < args.map_max_trajectories_per_cluster:
            balanced_uids.extend([
                uid
                for uid in cluster_rows["TRAJECTORY_UID"].tolist()
                if uid not in selected_uid_set
            ])

        cluster_uids = balanced_uids[:args.map_max_trajectories_per_cluster]
        selected_uids.extend(cluster_uids)

    if not selected_uids:
        print("Carte ignorée : aucun trajet sélectionné.")
        return

    map_points = points_df[points_df["TRAJECTORY_UID"].isin(selected_uids)].copy()

    if map_points.empty:
        print("Carte ignorée : aucun point trouvé pour les clusters sélectionnés.")
        return

    sort_columns = ["TRAJECTORY_UID"]
    if "POINT_INDEX" in map_points.columns:
        sort_columns.append("POINT_INDEX")
    elif "DATETIME" in map_points.columns:
        sort_columns.append("DATETIME")

    map_points = map_points.sort_values(sort_columns)

    cluster_by_uid = dict(
        selected_clusters[["TRAJECTORY_UID", "CLUSTER_ID"]].values.tolist()
    )
    summary_by_cluster = summary_df.set_index("CLUSTER_ID").to_dict("index")

    center = [
        map_points["LATITUDE"].mean(),
        map_points["LONGITUDE"].mean()
    ]

    map_object = folium.Map(
        location=center,
        zoom_start=10,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    folium.TileLayer("CartoDB positron", name="Fond clair").add_to(map_object)
    folium.TileLayer("CartoDB dark_matter", name="Fond sombre").add_to(map_object)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(map_object)

    cluster_groups = {}

    for index, cluster_id in enumerate(top_cluster_ids):
        summary = summary_by_cluster.get(cluster_id, {})
        label = (
            f"Cluster {cluster_id} "
            f"({int(summary.get('N_TRAJECTORIES', 0))} trajets, "
            f"{int(summary.get('N_USERS', 0))} users)"
        )
        group = folium.FeatureGroup(name=label, show=index < 8)
        group.add_to(map_object)
        cluster_groups[cluster_id] = group

    for trajectory_uid, trajectory_df in tqdm(
        map_points.groupby("TRAJECTORY_UID", sort=False),
        desc="Ajout des clusters à la carte",
        unit="trajet"
    ):
        cluster_id = cluster_by_uid.get(trajectory_uid)

        if cluster_id not in cluster_groups:
            continue

        points = trajectory_df[["LATITUDE", "LONGITUDE"]].values.tolist()

        if len(points) < 2:
            continue

        cluster_rank = top_cluster_ids.index(cluster_id)
        color = COLORS[cluster_rank % len(COLORS)]
        summary = summary_by_cluster.get(cluster_id, {})
        tooltip = (
            f"Cluster {cluster_id} | {trajectory_uid} | "
            f"{int(summary.get('N_TRAJECTORIES', 0))} trajets | "
            f"{int(summary.get('N_USERS', 0))} utilisateurs"
        )

        folium.PolyLine(
            points[::args.map_point_step],
            color=color,
            weight=3,
            opacity=0.55,
            tooltip=tooltip
        ).add_to(cluster_groups[cluster_id])

    Fullscreen().add_to(map_object)
    MiniMap(toggle_display=True, position="bottomright").add_to(map_object)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coordonnées",
        num_digits=5
    ).add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)

    map_object.fit_bounds([
        [map_points["LATITUDE"].min(), map_points["LONGITUDE"].min()],
        [map_points["LATITUDE"].max(), map_points["LONGITUDE"].max()]
    ])

    total_displayed = map_points["TRAJECTORY_UID"].nunique()
    panel_html = f"""
{{% macro html(this, kwargs) %}}
<style>
#cluster-panel {{
    position: fixed;
    top: 20px;
    left: 20px;
    width: 330px;
    z-index: 9999;
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    padding: 16px;
    font-family: Arial, sans-serif;
    color: #0f172a;
}}
#cluster-panel h2 {{
    margin: 0 0 10px;
    font-size: 17px;
}}
#cluster-panel p {{
    margin: 6px 0;
    color: #475569;
    font-size: 13px;
    line-height: 1.45;
}}
</style>
<div id="cluster-panel">
    <h2>Clusters de trajectoires</h2>
    <p><b>{len(top_cluster_ids)}</b> clusters affichés.</p>
    <p><b>{total_displayed}</b> trajets dessinés.</p>
    <p>Minimum carte : <b>{args.map_min_users}</b> utilisateurs par cluster</p>
    <p>Grille : <b>{args.cell_size_m} m</b></p>
    <p>Similarité Jaccard : <b>{args.jaccard_threshold}</b></p>
</div>
{{% endmacro %}}
"""

    panel = MacroElement()
    panel._template = Template(panel_html)
    map_object.get_root().add_child(panel)

    output_path = os.path.abspath(args.map_output)
    map_object.save(output_path)
    print(f"Carte clusters        : {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clustering de trajectoires GPS raffinées par grille géographique."
    )
    parser.add_argument("--input", default=REFINED_TRAJECTORIES_FILE)
    parser.add_argument("--individuals", default=INDIVIDUALS_FILE)
    parser.add_argument("--features-output", default=TRAJECTORY_FEATURES_FILE)
    parser.add_argument("--general-output", default=GENERAL_CLUSTERS_FILE)
    parser.add_argument("--general-summary-output", default=GENERAL_SUMMARY_FILE)
    parser.add_argument("--profile-output", default=PROFILE_CLUSTERS_FILE)
    parser.add_argument("--profile-summary-output", default=PROFILE_SUMMARY_FILE)
    parser.add_argument("--map-output", default=CLUSTER_MAP_FILE)
    parser.add_argument("--cell-size-m", type=int, default=DEFAULT_CELL_SIZE_M)
    parser.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD)
    parser.add_argument("--min-shared-cells", type=int, default=DEFAULT_MIN_SHARED_CELLS)
    parser.add_argument("--profile-min-trajectories", type=int, default=10)
    parser.add_argument("--criteria", nargs="+", default=DEFAULT_PROFILE_CRITERIA)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--map-top-clusters", type=int, default=20)
    parser.add_argument("--map-min-cluster-size", type=int, default=2)
    parser.add_argument("--map-min-users", type=int, default=2)
    parser.add_argument("--map-max-trajectories-per-cluster", type=int, default=30)
    parser.add_argument("--map-max-trajectories-per-user", type=int, default=2)
    parser.add_argument("--map-point-step", type=int, default=2)
    parser.add_argument(
        "--map-rank-by",
        choices=["users", "trajectories"],
        default="users",
        help="Priorite de selection des clusters sur la carte."
    )
    parser.add_argument(
        "--no-od-filter",
        action="store_true",
        help="Compare les trajets sans imposer même cellule de départ et d'arrivée."
    )
    parser.add_argument(
        "--general-only",
        action="store_true",
        help="Ne lance que le clustering général."
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Regenere seulement la carte depuis les CSV de clusters existants."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force le recalcul des features et clusters meme si les CSV existent deja."
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Ne génère pas la carte HTML des clusters."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n===== TRAJECTORY CLUSTERING =====\n")
    print(f"Entrée trajectoires : {os.path.abspath(args.input)}")
    print(f"Taille cellule      : {args.cell_size_m} m")
    print(f"Seuil Jaccard       : {args.jaccard_threshold}")
    print(f"Cellules communes   : {args.min_shared_cells}")
    print(f"Filtre origine/dest : {not args.no_od_filter}\n")

    general_output_exists = os.path.exists(args.general_output)
    general_summary_exists = os.path.exists(args.general_summary_output)
    can_reuse_general = general_output_exists and general_summary_exists

    points_df = load_refined_points(args.input, max_trajectories=args.max_trajectories)
    print(f"Points chargés      : {len(points_df)}")
    print(f"Trajets chargés     : {points_df['TRAJECTORY_UID'].nunique()}\n")

    if (
        args.map_only
        or (
            can_reuse_general
            and not args.recompute
            and not args.no_map
            and not args.general_only
        )
    ):
        if not general_output_exists:
            raise FileNotFoundError(args.general_output)
        if not general_summary_exists:
            raise FileNotFoundError(args.general_summary_output)

        general_clusters = pd.read_csv(args.general_output)
        general_summary = pd.read_csv(args.general_summary_output)
        print("CSV clusters existants trouves.")
        print("Generation de la carte uniquement, sans recalcul des features/clusters.")
        build_cluster_map(points_df, general_clusters, general_summary, args)
        print("\nTerminÃ©.")
        return

    features_df = build_trajectory_features(points_df, args.cell_size_m)
    features_df.to_csv(args.features_output, index=False)
    print(f"Features trajets    : {len(features_df)}")
    print(f"Fichier features    : {os.path.abspath(args.features_output)}\n")

    general_clusters, general_summary = run_general_clustering(features_df, args)
    print("===== CLUSTERING GÉNÉRAL =====")
    print(f"Trajets clusterisés : {len(general_clusters)}")
    print(f"Clusters            : {general_clusters['CLUSTER_ID'].nunique()}")
    print(f"Sortie              : {os.path.abspath(args.general_output)}")
    print(f"Résumé              : {os.path.abspath(args.general_summary_output)}\n")

    if not args.no_map:
        build_cluster_map(points_df, general_clusters, general_summary, args)
        print()

    if not args.general_only:
        individuals_df = load_individuals(args.individuals)
        profile_clusters, profile_summary = run_profile_clustering(
            features_df,
            individuals_df,
            args
        )

        print("===== CLUSTERING PAR CRITÈRE =====")
        print(f"Critères            : {', '.join(args.criteria)}")
        print(f"Lignes clusterisées : {len(profile_clusters)}")
        print(f"Sortie              : {os.path.abspath(args.profile_output)}")
        print(f"Résumé              : {os.path.abspath(args.profile_summary_output)}\n")

    print("Terminé.")


if __name__ == "__main__":
    main()
