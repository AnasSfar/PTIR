from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "results" / "focused_clustering_output" / "focused_general_clusters.csv"
OUT_DIR = BASE / "results" / "jaccard_threshold_study"
THRESHOLDS = [0.25, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60]
MIN_SHARED_CELLS = 5
PAIR_MIN_THRESHOLD = min(THRESHOLDS)


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def load_features() -> list[dict]:
    rows = []
    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "uid": row["TRAJECTORY_UID"],
                    "user": row["USER_ID"],
                    "cells": set(row["CELL_SET"].split()),
                    "n_cells": int(float(row["N_CELLS"])),
                }
            )
    return rows


def jaccard(left: set[str], right: set[str]) -> tuple[float, int]:
    shared = len(left & right)
    if shared == 0:
        return 0.0, 0
    return shared / len(left | right), shared


def build_candidate_pairs(features: list[dict]) -> list[tuple[float, str, str]]:
    pairs = []
    total = len(features)
    for i, left in enumerate(features):
        if i % 250 == 0:
            print(f"Comparaison {i:,}/{total:,}")
        for right in features[i + 1 :]:
            score, shared = jaccard(left["cells"], right["cells"])
            if shared >= MIN_SHARED_CELLS and score >= PAIR_MIN_THRESHOLD:
                pairs.append((score, left["uid"], right["uid"]))
    pairs.sort(reverse=True)
    return pairs


def summarize_for_threshold(features: list[dict], pairs: list[tuple[float, str, str]], threshold: float) -> dict:
    ids = [row["uid"] for row in features]
    users_by_uid = {row["uid"]: row["user"] for row in features}
    uf = UnionFind(ids)
    kept_edges = 0
    for score, left, right in pairs:
        if score < threshold:
            break
        uf.union(left, right)
        kept_edges += 1

    cluster_to_uids: dict[str, list[str]] = defaultdict(list)
    for uid in ids:
        cluster_to_uids[uf.find(uid)].append(uid)

    summaries = []
    for uids in cluster_to_uids.values():
        users = {users_by_uid[uid] for uid in uids}
        summaries.append({"n_trajectories": len(uids), "n_users": len(users)})

    collective = [
        row for row in summaries
        if row["n_users"] >= 2 and row["n_trajectories"] >= 3
    ]
    trajectories_in_collective = sum(row["n_trajectories"] for row in collective)
    largest = max(summaries, key=lambda row: row["n_trajectories"])
    largest_collective = max(collective, key=lambda row: row["n_trajectories"]) if collective else {"n_trajectories": 0, "n_users": 0}

    return {
        "threshold": threshold,
        "candidate_edges": kept_edges,
        "clusters_total": len(summaries),
        "singletons": sum(1 for row in summaries if row["n_trajectories"] == 1),
        "collective_clusters": len(collective),
        "trajectories_in_collective": trajectories_in_collective,
        "collective_coverage_pct": round(100 * trajectories_in_collective / len(features), 2),
        "largest_cluster_trajectories": largest["n_trajectories"],
        "largest_cluster_users": largest["n_users"],
        "largest_collective_trajectories": largest_collective["n_trajectories"],
        "largest_collective_users": largest_collective["n_users"],
        "avg_collective_users": round(sum(row["n_users"] for row in collective) / len(collective), 2) if collective else 0,
    }


def save_csv(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "jaccard_threshold_sensitivity.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {path}")


def save_plot(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    metrics = [
        ("collective_clusters", "Clusters collectifs"),
        ("trajectories_in_collective", "Trajectoires dans clusters collectifs"),
        ("largest_cluster_trajectories", "Taille du plus gros cluster"),
        ("singletons", "Clusters isolés"),
    ]
    for ax, (key, title) in zip(axes.flatten(), metrics):
        ax.plot(thresholds, [row[key] for row in rows], marker="o", color="#2563eb", lw=2)
        ax.axvline(0.42, color="#ef4444", ls="--", lw=1.8)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Seuil Jaccard")
        ax.grid(color="#d1d5db", alpha=0.7)
    fig.suptitle(
        "Etude de sensibilité du seuil Jaccard (min. 5 cellules communes)",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "jaccard_threshold_sensitivity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {path}")


def save_report_curve(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    selected = next(row for row in rows if abs(row["threshold"] - 0.42) < 1e-9)

    fig, ax1 = plt.subplots(figsize=(12.5, 6.7))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        thresholds,
        [row["collective_clusters"] for row in rows],
        marker="o",
        lw=2.8,
        color="#2563eb",
        label="Clusters collectifs",
    )
    line2 = ax1.plot(
        thresholds,
        [row["largest_cluster_trajectories"] for row in rows],
        marker="s",
        lw=2.8,
        color="#f97316",
        label="Taille du plus gros cluster",
    )
    line3 = ax2.plot(
        thresholds,
        [row["singletons"] for row in rows],
        marker="^",
        lw=2.8,
        color="#64748b",
        label="Trajectoires isolées",
    )

    ax1.axvline(0.42, color="#ef4444", ls="--", lw=2.2)
    ax1.scatter([0.42], [selected["collective_clusters"]], s=130, color="#ef4444", zorder=5)
    ax1.scatter([0.42], [selected["largest_cluster_trajectories"]], s=130, color="#ef4444", zorder=5)
    ax2.scatter([0.42], [selected["singletons"]], s=130, color="#ef4444", zorder=5)

    ax1.annotate(
        "Seuil retenu = 0,42\n44 clusters collectifs\nplus gros cluster : 71 trajets",
        xy=(0.42, selected["largest_cluster_trajectories"]),
        xytext=(0.455, 255),
        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.8),
        bbox=dict(boxstyle="round,pad=0.45", fc="#fef2f2", ec="#ef4444", lw=1.2),
        fontsize=10.5,
        color="#111827",
    )

    ax1.annotate(
        "Seuil trop faible :\ncluster géant",
        xy=(0.25, rows[0]["largest_cluster_trajectories"]),
        xytext=(0.285, 760),
        arrowprops=dict(arrowstyle="->", color="#f97316", lw=1.6),
        fontsize=10,
        color="#9a3412",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec="#f97316", lw=1),
    )

    ax2.annotate(
        "Seuil trop fort :\nbeaucoup d'isolés",
        xy=(0.60, rows[-1]["singletons"]),
        xytext=(0.505, 930),
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.6),
        fontsize=10,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8fafc", ec="#64748b", lw=1),
    )

    ax1.set_title(
        "Choix du seuil Jaccard par étude de sensibilité",
        fontsize=17,
        weight="bold",
        pad=16,
    )
    ax1.set_xlabel("Seuil de similarité Jaccard", fontsize=12)
    ax1.set_ylabel("Clusters collectifs / taille du plus gros cluster", fontsize=11, color="#111827")
    ax2.set_ylabel("Nombre de trajectoires isolées", fontsize=11, color="#64748b")

    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([f"{value:.2f}" for value in thresholds], rotation=0)
    ax1.grid(color="#d1d5db", alpha=0.75)
    ax1.set_ylim(0, 980)
    ax2.set_ylim(420, 1100)

    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower left", frameon=True, framealpha=0.95)

    ax1.text(
        0.5,
        -0.17,
        "Paramètre fixe dans tous les tests : au moins 5 cellules communes entre deux trajectoires.",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )

    path = OUT_DIR / "jaccard_threshold_report_curve.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Report curve: {path}")


def save_threshold_choice_schema(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.axis("off")

    thresholds = [row["threshold"] for row in rows]
    xs = [0.06 + i * (0.88 / (len(thresholds) - 1)) for i in range(len(thresholds))]
    y = 0.60

    ax.plot([xs[0], xs[-1]], [y, y], color="#94a3b8", lw=3, transform=ax.transAxes, zorder=1)

    for x, row in zip(xs, rows):
        threshold = row["threshold"]
        is_selected = abs(threshold - 0.42) < 1e-9
        if threshold < 0.40:
            color = "#f97316"
            label = "trop large"
        elif threshold > 0.50:
            color = "#64748b"
            label = "trop strict"
        elif is_selected:
            color = "#ef4444"
            label = "retenu"
        else:
            color = "#2563eb"
            label = "acceptable"

        radius = 0.019 if not is_selected else 0.029
        ax.add_patch(
            plt.Circle((x, y), radius, transform=ax.transAxes, fc=color, ec="white", lw=2.2, zorder=4)
        )
        ax.text(
            x,
            y + 0.065,
            f"{threshold:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10 if not is_selected else 12,
            weight="bold" if is_selected else "normal",
            color="#0f172a",
        )
        ax.text(
            x,
            y - 0.055,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color=color,
            weight="bold" if is_selected else "normal",
        )

        if threshold in {0.25, 0.35, 0.42, 0.50, 0.60}:
            box_y = 0.18 if threshold != 0.42 else 0.10
            box_h = 0.22 if threshold != 0.42 else 0.30
            box_w = 0.14 if threshold != 0.42 else 0.18
            ax.add_patch(
                plt.Rectangle(
                    (x - box_w / 2, box_y),
                    box_w,
                    box_h,
                    transform=ax.transAxes,
                    fc="#fff7ed" if threshold < 0.40 else ("#fef2f2" if is_selected else "#f8fafc"),
                    ec=color,
                    lw=1.4,
                    zorder=2,
                )
            )
            ax.text(
                x,
                box_y + box_h - 0.045,
                f"seuil {threshold:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#0f172a",
                zorder=5,
            )
            lines = [
                f"{row['collective_clusters']} clusters coll.",
                f"{row['trajectories_in_collective']} traj. regroupes",
                f"max: {row['largest_cluster_trajectories']} traj.",
                f"{row['singletons']} isoles",
            ]
            for i, line in enumerate(lines):
                ax.text(
                    x,
                    box_y + box_h - 0.085 - i * 0.04,
                    line,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7.8 if not is_selected else 8.6,
                    color="#334155",
                    zorder=5,
                )

    ax.text(
        0.5,
        0.92,
        "Choix du seuil Jaccard : comparaison des valeurs testées",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.84,
        "Critère conservé pour tous les tests : au moins 5 cellules communes",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )
    ax.text(
        0.20,
        0.74,
        "Seuil faible : beaucoup de trajets fusionnés\nmais clusters trop larges",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#9a3412",
    )
    ax.text(
        0.50,
        0.74,
        "0,42 : compromis choisi\nclusters collectifs sans cluster géant",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color="#b91c1c",
    )
    ax.text(
        0.81,
        0.74,
        "Seuil fort : résultats plus stricts\nmais beaucoup de trajets isolés",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )

    path = OUT_DIR / "jaccard_threshold_choice_schema.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Schema: {path}")


def save_report_curve(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    selected = next(row for row in rows if abs(row["threshold"] - 0.42) < 1e-9)

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.4), sharex=True)
    panels = [
        {
            "key": "collective_clusters",
            "title": "Nombre de clusters collectifs",
            "color": "#2563eb",
            "marker": "o",
            "note": "objectif : garder des groupes interpretables",
        },
        {
            "key": "largest_cluster_trajectories",
            "title": "Taille du plus gros cluster (en trajectoires)",
            "color": "#f97316",
            "marker": "s",
            "note": "trop haut = fusion excessive de trajets differents",
        },
        {
            "key": "singletons",
            "title": "Trajectoires isolees",
            "color": "#64748b",
            "marker": "^",
            "note": "trop haut = seuil trop strict",
        },
    ]

    for ax, panel in zip(axes, panels):
        values = [row[panel["key"]] for row in rows]
        selected_value = selected[panel["key"]]
        color = panel["color"]

        ax.plot(thresholds, values, marker=panel["marker"], lw=2.7, color=color)
        ax.axvline(0.42, color="#ef4444", ls="--", lw=1.9)
        ax.scatter([0.42], [selected_value], s=95, color="#ef4444", zorder=5)
        ax.set_title(panel["title"], loc="left", fontsize=11.5, weight="bold", color="#111827")
        ax.text(
            0.42,
            selected_value,
            f"  {selected_value}",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#b91c1c",
            weight="bold",
        )
        ax.grid(color="#d1d5db", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_min, y_max = min(values), max(values)
        margin = max((y_max - y_min) * 0.18, 5)
        ax.set_ylim(max(0, y_min - margin), y_max + margin)

    axes[-1].set_xlabel("Seuil de similarite Jaccard", fontsize=12)
    axes[-1].set_xticks(thresholds)
    axes[-1].set_xticklabels([f"{value:.2f}" for value in thresholds], rotation=0)

    fig.suptitle(
        "Choix du seuil Jaccard par etude de sensibilite",
        fontsize=17,
        weight="bold",
        y=0.98,
    )
    fig.text(0.20, 0.935, "Seuil faible : clusters trop larges", ha="center", fontsize=10, color="#9a3412")
    fig.text(0.50, 0.935, "0,42 : compromis retenu", ha="center", fontsize=10, color="#b91c1c", weight="bold")
    fig.text(0.80, 0.935, "Seuil fort : beaucoup d'isoles", ha="center", fontsize=10, color="#334155")
    fig.text(
        0.5,
        0.02,
        "Parametre fixe dans tous les tests : au moins 5 cellules communes entre deux trajectoires.",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=[0.035, 0.055, 0.985, 0.94])

    path = OUT_DIR / "jaccard_threshold_report_curve.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Report curve: {path}")


def save_report_curve(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    selected = next(row for row in rows if abs(row["threshold"] - 0.42) < 1e-9)
    series = [
        ("collective_clusters", "Clusters collectifs", "#2563eb", "o"),
        ("largest_cluster_trajectories", "Taille du plus gros cluster", "#f97316", "s"),
        ("singletons", "Trajectoires isolees", "#64748b", "^"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 6.8))

    for key, label, color, marker in series:
        raw_values = [row[key] for row in rows]
        min_value = min(raw_values)
        max_value = max(raw_values)
        normalized = [
            100 * (value - min_value) / (max_value - min_value)
            if max_value != min_value else 0
            for value in raw_values
        ]
        selected_index = thresholds.index(0.42)

        ax.plot(thresholds, normalized, marker=marker, lw=2.9, color=color, label=label)
        ax.scatter([0.42], [normalized[selected_index]], s=110, color="#ef4444", zorder=5)
        ax.text(
            0.422,
            normalized[selected_index],
            f" {selected[key]}",
            va="center",
            ha="left",
            fontsize=9.5,
            color=color,
            weight="bold",
        )
        ax.text(
            thresholds[-1] + 0.006,
            normalized[-1],
            label,
            va="center",
            ha="left",
            fontsize=9.5,
            color=color,
        )

    ax.axvspan(0.25, 0.38, color="#fff7ed", alpha=0.65, zorder=0)
    ax.axvspan(0.40, 0.45, color="#fef2f2", alpha=0.65, zorder=0)
    ax.axvspan(0.50, 0.60, color="#f8fafc", alpha=0.8, zorder=0)
    ax.axvline(0.42, color="#ef4444", ls="--", lw=2.2)

    ax.text(0.315, 105, "seuil faible : clusters trop larges", ha="center", fontsize=9.5, color="#9a3412")
    ax.text(0.425, 105, "0,42 : compromis", ha="center", fontsize=9.5, color="#b91c1c", weight="bold")
    ax.text(0.55, 105, "seuil fort : beaucoup d'isoles", ha="center", fontsize=9.5, color="#334155")

    ax.set_title("Choix du seuil Jaccard par etude de sensibilite", fontsize=17, weight="bold", pad=18)
    ax.set_xlabel("Seuil de similarite Jaccard", fontsize=12)
    ax.set_ylabel("Valeur normalisee pour comparer les tendances", fontsize=11)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{value:.2f}" for value in thresholds])
    ax.set_ylim(-8, 112)
    ax.set_xlim(0.235, 0.645)
    ax.grid(color="#d1d5db", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    ax.text(
        0.5,
        -0.16,
        "Les courbes sont normalisees pour etre lisibles sur un meme graphe. Les nombres affiches a 0,42 sont les valeurs reelles.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#475569",
    )
    ax.text(
        0.5,
        -0.22,
        "Parametre fixe : au moins 5 cellules communes entre deux trajectoires.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#475569",
    )

    path = OUT_DIR / "jaccard_threshold_report_curve.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Report curve: {path}")


def main() -> None:
    features = load_features()
    print(f"Trajectoires: {len(features):,}")
    pairs = build_candidate_pairs(features)
    print(f"Paires candidates >= {PAIR_MIN_THRESHOLD}: {len(pairs):,}")
    rows = [summarize_for_threshold(features, pairs, threshold) for threshold in THRESHOLDS]
    save_csv(rows)
    save_plot(rows)
    save_report_curve(rows)
    save_threshold_choice_schema(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
