from __future__ import annotations

import csv
import html
import math
import os
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
OUT = BASE / "PTIR Anas - suite clustering.docx"
ASSET_DIR = BASE / "report_assets_suite"
FOCUSED_DIR = BASE / "focused_clustering_output"


PALETTE = {
    "blue": "#1f6feb",
    "green": "#2da44e",
    "orange": "#fb8500",
    "red": "#d1242f",
    "purple": "#8250df",
    "gray": "#57606a",
    "light": "#f6f8fa",
    "dark": "#24292f",
}

MAP_COLORS = [
    "#22d3ee",
    "#f97316",
    "#a78bfa",
    "#2dd4bf",
    "#60a5fa",
    "#f43f5e",
    "#84cc16",
    "#eab308",
]


def add_static_map_chrome(
    ax,
    title: str,
    metrics: list[tuple[str, str]],
    layers: list[str],
    profile: bool = False,
) -> None:
    """Draw a static version of the Folium UI used in the existing HTML maps."""
    ax.set_facecolor("#eef2f6")
    ax.grid(color="#cfd8e3", linewidth=0.7, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    panel_x, panel_y, panel_w, panel_h = 0.035, 0.60, 0.39, 0.34
    panel = plt.Rectangle(
        (panel_x, panel_y),
        panel_w,
        panel_h,
        transform=ax.transAxes,
        fc="#0f172a",
        ec="#64748b",
        lw=1.0,
        alpha=0.94,
        zorder=20,
    )
    ax.add_patch(panel)
    ax.add_patch(
        plt.Rectangle(
            (panel_x, panel_y + panel_h - 0.075),
            panel_w,
            0.075,
            transform=ax.transAxes,
            fc="#123447",
            ec="none",
            alpha=0.95,
            zorder=21,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (panel_x + panel_w * 0.50, panel_y + panel_h - 0.075),
            panel_w * 0.50,
            0.075,
            transform=ax.transAxes,
            fc="#3b2a1d",
            ec="none",
            alpha=0.82,
            zorder=22,
        )
    )
    ax.text(
        panel_x + 0.02,
        panel_y + panel_h - 0.04,
        title,
        transform=ax.transAxes,
        color="#f8fafc",
        fontsize=10,
        weight="bold",
        va="center",
        zorder=23,
    )
    row_y = panel_y + panel_h - 0.11
    for label, value in metrics:
        ax.text(panel_x + 0.025, row_y, label, transform=ax.transAxes, color="#aab8cc", fontsize=8.2, va="center", zorder=23)
        ax.text(panel_x + panel_w - 0.025, row_y, value, transform=ax.transAxes, color="#f8fafc", fontsize=8.2, va="center", ha="right", family="monospace", zorder=23)
        ax.plot([panel_x + 0.02, panel_x + panel_w - 0.02], [row_y - 0.026, row_y - 0.026], transform=ax.transAxes, color="#334155", lw=0.8, zorder=23)
        row_y -= 0.052

    if profile:
        ax.text(panel_x + 0.025, panel_y + 0.047, "Critere", transform=ax.transAxes, color="#aab8cc", fontsize=7.8, zorder=23)
        ax.text(panel_x + 0.20, panel_y + 0.047, "Valeur", transform=ax.transAxes, color="#aab8cc", fontsize=7.8, zorder=23)
        ax.add_patch(plt.Rectangle((panel_x + 0.025, panel_y + 0.015), 0.14, 0.027, transform=ax.transAxes, fc="#0f172a", ec="#475569", lw=0.8, zorder=23))
        ax.add_patch(plt.Rectangle((panel_x + 0.20, panel_y + 0.015), 0.17, 0.027, transform=ax.transAxes, fc="#0f172a", ec="#475569", lw=0.8, zorder=23))
        ax.text(panel_x + 0.033, panel_y + 0.028, "SEX", transform=ax.transAxes, color="#f8fafc", fontsize=7.2, va="center", zorder=24)
        ax.text(panel_x + 0.208, panel_y + 0.028, "Man", transform=ax.transAxes, color="#f8fafc", fontsize=7.2, va="center", zorder=24)

    layer_w = 0.30
    layer_h = min(0.08 + 0.04 * len(layers), 0.38)
    lx, ly = 0.66, 0.07
    ax.add_patch(plt.Rectangle((lx, ly), layer_w, layer_h, transform=ax.transAxes, fc="white", ec="#94a3b8", lw=0.8, alpha=0.95, zorder=20))
    ax.text(lx + 0.015, ly + layer_h - 0.03, "Couches", transform=ax.transAxes, color="#111827", fontsize=8.5, weight="bold", zorder=21)
    for i, layer in enumerate(layers[:7]):
        y = ly + layer_h - 0.065 - 0.037 * i
        ax.add_patch(plt.Rectangle((lx + 0.018, y - 0.01), 0.018, 0.018, transform=ax.transAxes, fc="white", ec="#64748b", lw=0.7, zorder=21))
        ax.text(lx + 0.047, y, layer, transform=ax.transAxes, color="#111827", fontsize=7.2, va="center", zorder=21)


def pad_limits(values: list[float], pct: float = 0.08) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-6)
    return lo - span * pct, hi + span * pct


def read_csv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(row)
    return rows


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def save_pipeline_schema(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.axis("off")
    steps = [
        ("Trajectoires\nnettoyees", "CSV par trajet"),
        ("Grille spatiale", "cellules 350 m"),
        ("Similarite", "Jaccard\nsur cellules"),
        ("Clusters", "trajets proches"),
        ("Segments\nponderes", "cellules 90 m"),
        ("Profils", "age, sexe,\nmode, voiture"),
    ]
    xs = [0.08, 0.25, 0.42, 0.59, 0.76, 0.92]
    y = 0.58
    for i, ((title, sub), x) in enumerate(zip(steps, xs)):
        color = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["purple"], PALETTE["red"], PALETTE["gray"]][i]
        ax.add_patch(
            plt.Rectangle((x - 0.07, y - 0.18), 0.14, 0.34, fc="#ffffff", ec=color, lw=2)
        )
        ax.text(x, y + 0.045, title, ha="center", va="center", fontsize=10, weight="bold", color=PALETTE["dark"])
        ax.text(x, y - 0.105, sub, ha="center", va="center", fontsize=8.5, color=PALETTE["gray"])
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.085, y), xytext=(x + 0.08, y), arrowprops=dict(arrowstyle="->", lw=1.8, color="#8c959f"))
    ax.text(
        0.5,
        0.18,
        "Idee generale : transformer chaque trajet en objet comparable, puis faire ressortir les axes partages.",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["dark"],
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_grid_similarity_schema(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for x in range(11):
        ax.plot([x, x], [0, 6], color="#d0d7de", lw=0.8)
    for y in range(7):
        ax.plot([0, 10], [y, y], color="#d0d7de", lw=0.8)
    route_a = [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3), (7, 4), (8, 4)]
    route_b = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 3), (6, 3), (7, 3), (8, 3)]
    shared = set(route_a) & set(route_b)
    for x, y in shared:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#fff1b8", alpha=0.85))
    ax.plot([x + 0.5 for x, y in route_a], [y + 0.5 for x, y in route_a], "-o", lw=3, color=PALETTE["blue"], label="Trajet A")
    ax.plot([x + 0.5 for x, y in route_b], [y + 0.5 for x, y in route_b], "-o", lw=3, color=PALETTE["orange"], label="Trajet B")
    ax.legend(loc="upper left", frameon=False)
    ax.text(5, -0.55, "Cellules communes = intersection ; similarite Jaccard = intersection / union", ha="center", fontsize=10)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_weighted_segments_schema(path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    titles = [
        "1. Trajets complets differents",
        "2. Decoupage en segments",
        "3. Comptage des portions communes",
    ]
    for ax, title in zip(axes, titles):
        ax.set_aspect("equal")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")
        ax.set_title(title, fontsize=12, weight="bold", color="#0f172a", pad=10)
        for x in range(1, 10):
            ax.plot([x, x], [0.5, 6.5], color="#e5e7eb", lw=0.8, zorder=0)
        for y in range(1, 7):
            ax.plot([0.5, 9.5], [y, y], color="#e5e7eb", lw=0.8, zorder=0)

    traj_a = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (6, 4), (7, 5), (8, 6)]
    traj_b = [(1, 6), (2, 5), (3, 4), (4, 4), (5, 4), (6, 4), (7, 3), (8, 2)]
    traj_c = [(1, 3), (2, 3), (3, 3), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4)]
    routes = [(traj_a, "#22d3ee", "Utilisateur A"), (traj_b, "#f97316", "Utilisateur B"), (traj_c, "#a78bfa", "Utilisateur C")]

    for pts, color, label in routes:
        xs, ys = zip(*pts)
        axes[0].plot(xs, ys, "-o", color=color, lw=3, ms=4, label=label, solid_capstyle="round")
    axes[0].text(8.2, 6.05, "A", color="#22d3ee", fontsize=11, weight="bold")
    axes[0].text(8.2, 1.75, "B", color="#f97316", fontsize=11, weight="bold")
    axes[0].text(8.2, 4.25, "C", color="#a78bfa", fontsize=11, weight="bold")
    axes[0].text(
        5,
        0.35,
        "Departs/arrivees differents,\nportion centrale commune.",
        ha="center",
        va="top",
        fontsize=9,
        color="#475569",
    )

    for pts, color, _ in routes:
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            axes[1].plot([x1, x2], [y1, y2], color=color, lw=3, alpha=0.75, solid_capstyle="round")
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            axes[1].scatter([mx], [my], s=22, color="white", edgecolor=color, linewidth=1.4, zorder=3)
    axes[1].text(
        5,
        0.35,
        "Chaque trait entre deux cellules devient\nun segment que l'on peut compter.",
        ha="center",
        va="top",
        fontsize=9,
        color="#475569",
    )

    counter: Counter[tuple[tuple[int, int], tuple[int, int]]] = Counter()
    for pts, _, _ in routes:
        seen = set()
        for i in range(len(pts) - 1):
            edge = (pts[i], pts[i + 1])
            if edge not in seen:
                counter[edge] += 1
                seen.add(edge)

    for edge, count in counter.items():
        (x1, y1), (x2, y2) = edge
        if count == 1:
            color, lw, alpha = "#94a3b8", 1.5, 0.35
        elif count == 2:
            color, lw, alpha = "#f97316", 4.0, 0.85
        else:
            color, lw, alpha = "#ef4444", 6.5, 0.95
        axes[2].plot([x1, x2], [y1, y2], color="#020617", lw=lw + 1.2, alpha=0.20, solid_capstyle="round")
        axes[2].plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
        if count > 1:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            axes[2].text(
                mx,
                my + 0.28,
                f"{count} users",
                ha="center",
                va="center",
                fontsize=8,
                weight="bold",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cbd5e1", alpha=0.95),
            )
    axes[2].text(
        5,
        0.35,
        "Les segments les plus partages ressortent :\nce sont les axes/corridors interessants.",
        ha="center",
        va="top",
        fontsize=9,
        color="#475569",
    )

    fig.suptitle(
        "Principe des segments ponderes : trouver les portions communes meme si les trajets complets different",
        fontsize=14,
        weight="bold",
        color="#111827",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_cluster_plot(path: Path) -> None:
    clusters = read_csv(FOCUSED_DIR / "focused_general_clusters.csv")
    summary = read_csv(FOCUSED_DIR / "focused_general_summary.csv")
    top_ids = [row["CLUSTER_ID"] for row in sorted(summary, key=lambda r: as_int(r["N_USERS"]), reverse=True)[:5]]
    uid_to_cluster = {row["TRAJECTORY_UID"]: row["CLUSTER_ID"] for row in clusters if row["CLUSTER_ID"] in top_ids}
    wanted_uids = set(uid_to_cluster)

    points_by_uid: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with (FOCUSED_DIR / "focused_50_refined_trajectories.csv").open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["TRAJECTORY_UID"]
            if uid in wanted_uids and len(points_by_uid[uid]) < 280:
                points_by_uid[uid].append((as_float(row["LONGITUDE"]), as_float(row["LATITUDE"])))

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    colors = MAP_COLORS
    counts = Counter(uid_to_cluster.values())
    all_x: list[float] = []
    all_y: list[float] = []
    for rank, cluster_id in enumerate(top_ids):
        color = colors[rank % len(colors)]
        drawn = 0
        for uid, pts in points_by_uid.items():
            if uid_to_cluster[uid] != cluster_id or len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            all_x.extend(xs)
            all_y.extend(ys)
            ax.plot(xs, ys, color="#020617", alpha=0.22, lw=3.0, solid_capstyle="round", zorder=2)
            ax.plot(xs, ys, color=color, alpha=0.88, lw=1.7, solid_capstyle="round", zorder=3)
            drawn += 1
            if drawn >= 14:
                break
    ax.set_xlim(*pad_limits(all_x, 0.04))
    ax.set_ylim(*pad_limits(all_y, 0.04))
    add_static_map_chrome(
        ax,
        "Clusters focalises",
        [
            ("Utilisateurs", "100"),
            ("Clusters affichables", "10"),
            ("Trajets representatifs", "1 / cluster"),
            ("Segments caches", "1,299"),
        ],
        ["Trajets representatifs", "Segments ponderes - cluster 102", "Segments ponderes - cluster 78", "Segments ponderes - tous profils"],
    )
    legend_y = 0.52
    for rank, cluster_id in enumerate(top_ids[:5]):
        ax.plot([0.05, 0.10], [legend_y, legend_y], transform=ax.transAxes, color=colors[rank], lw=3, zorder=25)
        ax.text(0.115, legend_y, f"Cluster {cluster_id} ({counts[cluster_id]} trajets)", transform=ax.transAxes, color="#0f172a", fontsize=8, va="center", zorder=25)
        legend_y -= 0.04
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_weighted_segments_plot(path: Path) -> None:
    segments = read_csv(FOCUSED_DIR / "focused_general_weighted_segments.csv")
    segments = sorted(segments, key=lambda r: as_int(r["N_USERS"]), reverse=True)[:550]
    max_users = max(as_int(r["N_USERS"], 1) for r in segments) if segments else 1
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    all_x: list[float] = []
    all_y: list[float] = []
    for row in segments:
        users = as_int(row["N_USERS"])
        lw = 0.4 + 4.5 * (users / max_users)
        alpha = 0.18 + 0.72 * (users / max_users)
        xs = [as_float(row["FROM_LONGITUDE"]), as_float(row["TO_LONGITUDE"])]
        ys = [as_float(row["FROM_LATITUDE"]), as_float(row["TO_LATITUDE"])]
        all_x.extend(xs)
        all_y.extend(ys)
        ax.plot(
            xs,
            ys,
            color="#020617",
            lw=lw + 1.0,
            alpha=alpha * 0.30,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color="#f97316" if users < max_users * 0.75 else "#ef4444",
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=3,
        )
    ax.set_xlim(*pad_limits(all_x, 0.04))
    ax.set_ylim(*pad_limits(all_y, 0.04))
    add_static_map_chrome(
        ax,
        "Segments ponderes",
        [
            ("Segments visibles", f"{len(segments):,}"),
            ("Utilisateurs max", str(max_users)),
            ("Lecture", "epaisseur = poids"),
            ("Cellule segment", "90 m"),
        ],
        ["Segments ponderes - tous profils", "Segments ponderes - cluster 102", "Segments ponderes - cluster 78"],
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_profile_bars(path: Path) -> None:
    rows = read_csv(FOCUSED_DIR / "focused_profile_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["CRITERION"]].append(row)

    profile_points = []
    profile_records = []
    for criterion, color in [("SEX", "#2dd4bf"), ("DOMINANT_MODE", "#f97316"), ("AGE_GROUP", "#a78bfa")]:
        data = sorted(grouped.get(criterion, []), key=lambda r: as_int(r["N_TRAJECTORIES"]), reverse=True)[:2]
        for row in data:
            profile_records.append((criterion, row["VALUE"], as_int(row["N_USERS"]), as_int(row["N_TRAJECTORIES"]), color))

    clusters = read_csv(FOCUSED_DIR / "focused_general_clusters.csv")
    top_uids = [row["TRAJECTORY_UID"] for row in clusters[:16]]
    wanted = set(top_uids)
    by_uid: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with (FOCUSED_DIR / "focused_50_refined_trajectories.csv").open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["TRAJECTORY_UID"]
            if uid in wanted and len(by_uid[uid]) < 180:
                by_uid[uid].append((as_float(row["LONGITUDE"]), as_float(row["LATITUDE"])))

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    all_x: list[float] = []
    all_y: list[float] = []
    for i, (uid, pts) in enumerate(by_uid.items()):
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        all_x.extend(xs)
        all_y.extend(ys)
        color = MAP_COLORS[i % len(MAP_COLORS)]
        ax.plot(xs, ys, color="#020617", alpha=0.20, lw=3.0, solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, color=color, alpha=0.86, lw=1.8, solid_capstyle="round", zorder=3)
    ax.set_xlim(*pad_limits(all_x, 0.04))
    ax.set_ylim(*pad_limits(all_y, 0.04))
    add_static_map_chrome(
        ax,
        "Clusters par profil",
        [
            ("Lignes", str(len(profile_records))),
            ("Trajets", str(sum(r[3] for r in profile_records))),
            ("Utilisateurs max", str(max((r[2] for r in profile_records), default=0))),
            ("Lecture", "1 ligne / cluster"),
        ],
        ["AGE_GROUP = 35-44", "DOMINANT_MODE = WALKING", "SEX = Man", "SEX = Woman"],
        profile=True,
    )
    y = 0.51
    for criterion, value, users, trajs, color in profile_records[:6]:
        ax.plot([0.05, 0.10], [y, y], transform=ax.transAxes, color=color, lw=3, zorder=25)
        ax.text(0.115, y, f"{criterion}={value} | {users} u. | {trajs} traj.", transform=ax.transAxes, color="#0f172a", fontsize=7.6, va="center", zorder=25)
        y -= 0.037
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_table_image(path: Path) -> None:
    summary = read_csv(FOCUSED_DIR / "focused_general_summary.csv")
    top = sorted(summary, key=lambda r: as_int(r["N_USERS"]), reverse=True)[:8]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.axis("off")
    headers = ["Cluster", "Trajets", "Utilisateurs", "Pts moy.", "Cellules moy."]
    table_rows = [
        [
            r["CLUSTER_ID"],
            r["N_TRAJECTORIES"],
            r["N_USERS"],
            str(round(as_float(r["AVG_POINTS"]), 1)),
            str(round(as_float(r["AVG_CELLS"]), 1)),
        ]
        for r in top
    ]
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#0969da")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f6f8fa" if row % 2 else "white")
    ax.set_title("Principaux clusters detectes sur l'echantillon focalise", fontsize=12, weight="bold", pad=12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_assets() -> dict[str, Path]:
    ASSET_DIR.mkdir(exist_ok=True)
    paths = {
        "pipeline": ASSET_DIR / "schema_suite_pipeline.png",
        "grid": ASSET_DIR / "schema_grille_jaccard.png",
        "segments_schema": ASSET_DIR / "schema_segments_ponderes.png",
        "jaccard_vs_segments": ASSET_DIR / "schema_jaccard_vs_segments.png",
        "jaccard_study": BASE / "jaccard_threshold_study" / "jaccard_threshold_report_curve.png",
        "clusters": ASSET_DIR / "exemple_clusters.png",
        "segments": ASSET_DIR / "segments_ponderes.png",
        "profiles": ASSET_DIR / "profils_barres.png",
        "table": ASSET_DIR / "table_clusters.png",
    }
    save_pipeline_schema(paths["pipeline"])
    save_grid_similarity_schema(paths["grid"])
    save_weighted_segments_schema(paths["segments_schema"])
    real_general = ASSET_DIR / "map_focused_general_capture.png"
    real_segments = ASSET_DIR / "map_focused_segments_capture.png"
    real_profile = ASSET_DIR / "map_focused_profile_capture.png"
    if real_general.exists():
        paths["clusters"] = real_general
        paths["segments"] = real_segments if real_segments.exists() else real_general
    else:
        save_cluster_plot(paths["clusters"])
        save_weighted_segments_plot(paths["segments"])
    if real_profile.exists():
        paths["profiles"] = real_profile
    else:
        save_profile_bars(paths["profiles"])
    save_summary_table_image(paths["table"])
    return paths


class DocxBuilder:
    def __init__(self) -> None:
        self.body: list[str] = []
        self.rels: list[tuple[str, str]] = []
        self.media: list[tuple[str, Path]] = []
        self.next_rid = 1
        self.next_img = 1

    def add_paragraph(self, text: str = "", style: str | None = None, bold: bool = False, italic: bool = False) -> None:
        ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
        rpr = ""
        if bold or italic:
            rpr = "<w:rPr>" + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + "</w:rPr>"
        self.body.append(f"<w:p>{ppr}<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>")

    def add_bullet(self, text: str) -> None:
        self.body.append(
            "<w:p><w:pPr><w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"1\"/></w:numPr></w:pPr>"
            f"<w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"
        )

    def add_image(self, path: Path, width_in: float = 6.4, caption: str | None = None) -> None:
        rid = f"rId{self.next_rid}"
        self.next_rid += 1
        img_name = f"image{self.next_img}.png"
        self.next_img += 1
        self.rels.append((rid, f"media/{img_name}"))
        self.media.append((img_name, path))
        cx = int(width_in * 914400)
        cy = int(cx * 0.58)
        self.body.append(
            f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r><w:drawing>
    <wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0">
      <wp:extent cx="{cx}" cy="{cy}"/>
      <wp:docPr id="{self.next_img + 100}" name="{html.escape(img_name)}"/>
      <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
        <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
          <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:nvPicPr><pic:cNvPr id="0" name="{html.escape(img_name)}"/><pic:cNvPicPr/></pic:nvPicPr>
            <pic:blipFill><a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
            <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
          </pic:pic>
        </a:graphicData>
      </a:graphic>
    </wp:inline>
  </w:drawing></w:r>
</w:p>"""
        )
        if caption:
            self.add_paragraph(caption, style="Caption", italic=True)

    def page_break(self) -> None:
        self.body.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

    def build_document_xml(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
 <w:body>
  {''.join(self.body)}
  <w:sectPr>
   <w:pgSz w:w="11906" w:h="16838"/>
   <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>
  </w:sectPr>
 </w:body>
</w:document>"""

    def write(self, output: Path) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rel_entries = "\n".join(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
            for rid, target in self.rels
        )
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", CONTENT_TYPES)
            z.writestr("_rels/.rels", ROOT_RELS)
            z.writestr("docProps/core.xml", CORE_XML.format(now=now))
            z.writestr("docProps/app.xml", APP_XML)
            z.writestr("word/document.xml", self.build_document_xml())
            z.writestr("word/styles.xml", STYLES_XML)
            z.writestr("word/numbering.xml", NUMBERING_XML)
            z.writestr(
                "word/_rels/document.xml.rels",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rStyle" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
 <Relationship Id="rNum" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
 {rel_entries}
</Relationships>""",
            )
            for name, path in self.media:
                z.write(path, f"word/media/{name}")


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
 <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
 <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
 <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
 <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
 <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

CORE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>PTIR - suite clustering</dc:title>
 <dc:creator>Anas / Codex</dc:creator>
 <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
 <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
 <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
 <Application>Codex</Application>
</Properties>"""

STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
  <w:name w:val="Normal"/><w:qFormat/>
  <w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr>
  <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Title">
  <w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:spacing w:after="240"/></w:pPr>
  <w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="34"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Heading1">
  <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:spacing w:before="260" w:after="160"/></w:pPr>
  <w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="28"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Heading2">
  <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:spacing w:before="180" w:after="100"/></w:pPr>
  <w:rPr><w:b/><w:color w:val="365F91"/><w:sz w:val="24"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Caption">
  <w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:jc w:val="center"/><w:spacing w:after="220"/></w:pPr>
  <w:rPr><w:i/><w:color w:val="57606A"/><w:sz w:val="19"/></w:rPr>
 </w:style>
</w:styles>"""

NUMBERING_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:abstractNum w:abstractNumId="0">
  <w:lvl w:ilvl="0">
   <w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/>
   <w:lvlJc w:val="left"/>
   <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>
  </w:lvl>
 </w:abstractNum>
 <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""


def add_section(doc: DocxBuilder, title: str, paragraphs: list[str], bullets: list[str] | None = None) -> None:
    doc.add_paragraph(title, style="Heading1")
    for para in paragraphs:
        doc.add_paragraph(para)
    if bullets:
        for bullet in bullets:
            doc.add_bullet(bullet)


def main() -> None:
    assets = make_assets()
    general_summary = read_csv(FOCUSED_DIR / "focused_general_summary.csv")
    profile_summary = read_csv(FOCUSED_DIR / "focused_profile_summary.csv")
    selected_users = read_csv(FOCUSED_DIR / "selected_same_zone_users.csv")
    segments = read_csv(FOCUSED_DIR / "focused_general_weighted_segments.csv")

    n_users = len(selected_users)
    n_clusters = len(general_summary)
    n_segments = len(segments)
    max_users_cluster = max(as_int(r["N_USERS"]) for r in general_summary)
    max_traj_cluster = max(as_int(r["N_TRAJECTORIES"]) for r in general_summary)
    max_segment_users = max(as_int(r["N_USERS"]) for r in segments)
    total_profile_traj = sum(as_int(r["N_TRAJECTORIES"]) for r in profile_summary if r["CRITERION"] == "SEX")

    doc = DocxBuilder()
    doc.add_paragraph("PTIR – Analyse de la mobilité humaine (NetMob)", style="Title")
    doc.add_paragraph("Suite du rapport : clustering, corridors communs et analyse par profils", bold=True)
    doc.add_paragraph(
        "Dans le premier document, je me suis arrêté juste après la pipeline de nettoyage / segmentation. "
        "Cette suite reprend donc à partir des trajectoires déjà propres, avec l’objectif de montrer comment je passe de trajets individuels à une lecture plus collective des déplacements."
    )
    doc.add_image(assets["pipeline"], caption="Vue d'ensemble de la suite après la pipeline.")

    add_section(
        doc,
        "3. Passage des trajectoires aux objets comparables",
        [
            "Une fois les trajets nettoyés, le problème principal n’est plus le bruit GPS, mais la comparaison entre trajets. Deux personnes peuvent prendre le même axe avec un nombre de points différent, une vitesse différente, ou des points légèrement décalés. Comparer directement les coordonnées point par point serait donc trop fragile.",
            "J’ai donc transformé chaque trajectoire en une suite de cellules spatiales. L’idée est de projeter les points GPS sur une grille : au lieu de comparer chaque latitude / longitude exacte, on compare les zones traversées par le trajet. Cela rend la comparaison plus robuste aux petits écarts GPS.",
        ],
        [
            "Chaque trajet devient une liste de cellules traversées.",
            "La forme globale du trajet est conservée, mais les petits décalages sont absorbés.",
            "Deux trajets sont considérés proches s’ils partagent suffisamment de cellules.",
        ],
    )
    doc.add_image(assets["grid"], caption="Principe de comparaison : deux trajets différents peuvent partager les mêmes cellules centrales.")

    add_section(
        doc,
        "4. Mesure de similarité et clustering",
        [
            "Pour mesurer la proximité entre deux trajets, j’utilise une similarité de type Jaccard sur les ensembles de cellules. Concrètement, on regarde la proportion de cellules communes par rapport au nombre total de cellules visitées par les deux trajets.",
            "Le clustering regroupe ensuite les trajets qui dépassent un seuil de similarité. Dans le code, cela revient à relier les trajectoires proches, puis à fusionner les composantes connectées avec une structure Union-Find. Ce choix est simple à expliquer et pratique pour une première version, car il ne force pas un nombre de clusters à l’avance.",
        ],
        [
            "Entrée : trajectoires nettoyées et trimmées.",
            "Représentation : cellules de grille, ici principalement autour de 350 m pour le clustering.",
            "Similarité : part de cellules communes entre deux trajets.",
            "Sortie : un identifiant de cluster par trajectoire.",
        ],
    )
    doc.add_image(assets["table"], caption="Extrait des clusters les plus partagés dans l'expérience focalisée.")
    if assets["jaccard_study"].exists():
        doc.add_paragraph("Choix du seuil Jaccard", style="Heading2")
        doc.add_paragraph(
            "Pour choisir le seuil, j’ai testé plusieurs valeurs entre 0,25 et 0,60 en gardant la même contrainte minimale de 5 cellules communes. "
            "Un seuil trop faible fusionne trop de trajectoires dans de très gros clusters, ce qui devient difficile à interpréter. "
            "Un seuil trop élevé fragmente fortement les résultats et laisse beaucoup de trajectoires isolées. "
            "La valeur 0,42 correspond au compromis retenu : elle conserve des clusters collectifs exploitables sans produire un cluster géant qui mélange presque toute la zone."
        )
        doc.add_image(assets["jaccard_study"], caption="Étude de sensibilité utilisée pour justifier le choix du seuil Jaccard = 0,42.")

    add_section(
        doc,
        "5. Résultats sur un échantillon focalisé",
        [
            f"Pour éviter d’avoir une carte illisible à l’échelle de tout le dataset, j’ai aussi créé une version focalisée sur {n_users} utilisateurs situés dans une même zone. Cette étape sert surtout à tester si la méthode produit des groupes interprétables localement.",
            f"Sur cet échantillon, la sortie contient {n_clusters} clusters et {n_segments} segments pondérés. Le cluster le plus partagé atteint {max_users_cluster} utilisateurs, et le plus grand cluster contient {max_traj_cluster} trajectoires. Cela montre que certains axes reviennent réellement entre plusieurs personnes, même après suppression des zones trop personnelles.",
            "La carte HTML générée permet ensuite d’afficher les trajectoires représentatives des clusters, les variantes individuelles, et les segments les plus partagés. L’intérêt n’est pas seulement de dire que deux trajets se ressemblent, mais aussi de visualiser où ils se superposent.",
        ],
    )
    doc.add_image(assets["clusters"], caption="Aperçu hors fond de carte : plusieurs trajectoires d'un même cluster suivent des zones proches.")

    add_section(
        doc,
        "6. Segments pondérés et corridors communs",
        [
            "Le clustering Jaccard sert à regrouper les trajets qui se ressemblent dans leur ensemble. Mais pour trouver les axes vraiment fréquentés, on peut aussi regarder directement les portions de route. C’est l’intérêt des segments pondérés.",
            "Le principe est de découper les trajectoires en transitions entre cellules voisines. Ensuite, on compte combien d’utilisateurs passent par chaque transition. Une même transition répétée dans un même trajet ne doit pas artificiellement gonfler le score : l’objectif est de mesurer un partage entre utilisateurs, pas seulement une répétition locale.",
            f"Dans la sortie focalisée, le segment le plus fréquent est partagé par {max_segment_users} utilisateurs. Sur la carte, l’épaisseur du trait représente ce poids : plus le segment est épais, plus il est emprunté par plusieurs personnes.",
        ],
        [
            "Jaccard répond à : quels trajets complets se ressemblent ?",
            "Les segments pondérés répondent à : quelles portions de route sont souvent utilisées ?",
            "Les corridors reconstruits relient ensuite plusieurs segments successifs pour obtenir des axes plus continus.",
        ],
    )
    doc.add_image(assets["segments_schema"], caption="Schéma de principe : on compte les portions communes, même quand les trajets complets ne sont pas identiques.")
    doc.add_image(assets["segments"], caption="Segments pondérés : les traits épais correspondent aux portions les plus partagées.")

    add_section(
        doc,
        "7. Analyse par profils",
        [
            "Une fois les trajectoires communes détectées, j’ai commencé à relier les résultats aux informations des individus. L’objectif est de pouvoir comparer les mobilités selon plusieurs profils : âge, sexe, possession de voiture, abonnement Navigo ou mode dominant.",
            f"Dans le résumé de profils, on retrouve par exemple des groupes par âge, par mode dominant et par sexe. Pour le critère sexe, l'échantillon totalise {total_profile_traj} trajectoires analysées dans le fichier de synthèse. Ces filtres ne servent pas encore à conclure sociologiquement, mais ils rendent l’outil exploitable pour poser des questions plus précises.",
            "La carte des profils ajoute une petite interface avec deux menus : un critère et une valeur. On peut donc afficher les clusters représentatifs d’un groupe, par exemple les trajets associés au mode WALKING, au métro, à une tranche d’âge, etc.",
        ],
        [
            "AGE_GROUP : 18-24, 25-34, 35-44, etc.",
            "DOMINANT_MODE : WALKING, SUBWAY, BIKE.",
            "SEX : Man / Woman.",
            "NAVIGO_SUB, DRIVING_LICENCE et NB_CAR_GROUP pour les variables de transport.",
        ],
    )
    doc.add_image(assets["profiles"], caption="Exemple des volumes de trajectoires disponibles par profil.")

    add_section(
        doc,
        "8. Ce que les cartes produites permettent de montrer",
        [
            "À ce stade, le projet produit plusieurs cartes HTML : une carte de clusters généraux, une carte de segments partagés, une carte focalisée, et une carte par profils. Ce ne sont pas seulement des visualisations finales : elles servent aussi à contrôler si les paramètres choisis donnent des résultats cohérents.",
            "Par exemple, si un cluster regroupe surtout des trajets d’un seul utilisateur, il est moins intéressant pour l’analyse collective. À l’inverse, un cluster avec plusieurs utilisateurs et plusieurs trajectoires indique une zone de mobilité plus partagée. Les segments pondérés aident aussi à éviter de surestimer un trajet complet lorsque seule une portion est commune.",
        ],
        [
            "map_focused_general_clusters.html : lecture globale des clusters sur l’échantillon focalisé.",
            "map_focused_profile_clusters.html : lecture avec filtres de profils.",
            "focused_general_weighted_segments.csv : segments pondérés par utilisateurs et trajectoires.",
            "focused_general_summary.csv : résumé chiffré des clusters.",
        ],
    )

    add_section(
        doc,
        "9. Limites actuelles",
        [
            "Les résultats sont encourageants, mais ils restent exploratoires. Le principal point sensible est le choix des seuils : taille de grille, similarité minimale, nombre minimal d’utilisateurs, et taille des cellules pour les segments. Des seuils trop faibles créent beaucoup de petits groupes peu significatifs ; des seuils trop forts risquent de masquer des ressemblances réelles.",
            "Une autre limite est que la grille simplifie l’espace. C’est voulu pour rendre les trajectoires comparables, mais cela peut mélanger deux rues proches si la cellule est trop grande. À l’inverse, si les cellules sont trop petites, deux trajets similaires mais légèrement décalés ne se rencontrent plus.",
        ],
        [
            "Affiner les paramètres avec plusieurs zones tests.",
            "Distinguer les vrais trajets communs des allers-retours ou boucles locales.",
            "Améliorer la reconstruction des corridors continus à partir des segments.",
            "Comparer les profils seulement après avoir vérifié que les groupes sont suffisamment représentés.",
        ],
    )

    add_section(
        doc,
        "10. Bilan de l’avancement",
        [
            "Après la pipeline, j’ai donc ajouté une chaîne d’analyse qui transforme les trajectoires propres en clusters et en corridors communs. Le passage par la grille permet de comparer des trajets qui n’ont pas exactement les mêmes points GPS. Le clustering permet d’identifier des familles de trajets similaires, et les segments pondérés permettent d’isoler les portions réellement partagées.",
            "La partie profils commence à faire le lien entre mobilité spatiale et caractéristiques des usagers. Pour l’instant, je considère cette partie comme un outil d’exploration : elle montre que la pipeline complète fonctionne, mais les interprétations doivent encore être consolidées par des tests de paramètres et une analyse plus fine des cartes.",
        ],
    )

    if assets["jaccard_vs_segments"].exists():
        doc.add_paragraph("Schema recapitulatif : Jaccard et segments", style="Heading1")
        doc.add_image(assets["jaccard_vs_segments"], caption="Difference entre Jaccard et segments ponderes : le premier compare les trajets complets, les seconds localisent les portions communes.")

    doc.add_paragraph("Conclusion rapide", style="Heading1")
    doc.add_paragraph(
        "En résumé, je suis passé d’un fichier GPS brut à une chaîne complète : nettoyage, segmentation, trimming, sampling, puis clustering, segments communs et filtres de profils. La prochaine étape serait surtout de stabiliser les seuils et de choisir quelques exemples de zones / profils à analyser en détail."
    )

    doc.write(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
