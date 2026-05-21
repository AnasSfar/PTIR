from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


BASE = Path(__file__).resolve().parent
OUT = BASE / "report_assets_suite" / "schema_jaccard_vs_segments.png"


def box(ax, x, y, label, color, w=0.085, h=0.085):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="#334155", lw=1.5, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=12, weight="bold", color="#0f172a", zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#64748b", lw=2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=lw,
            color=color,
            shrinkA=14,
            shrinkB=14,
            zorder=2,
        )
    )


def path(ax, xs, y, labels, colors):
    for i in range(len(xs) - 1):
        arrow(ax, xs[i], y, xs[i + 1], y)
    for x, label, color in zip(xs, labels, colors):
        box(ax, x, y, label, color)


def panel(ax, title, subtitle, face):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.025, 0.045), 0.95, 0.88, facecolor=face, edgecolor="#cbd5e1", lw=1.4))
    ax.text(0.5, 0.86, title, ha="center", fontsize=14, weight="bold", color="#0f172a")
    ax.text(0.5, 0.79, subtitle, ha="center", fontsize=10.2, color="#475569")


def main():
    OUT.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1))
    fig.patch.set_facecolor("white")

    shared = "#a7f3d0"
    only_a = "#bfdbfe"
    only_b = "#fed7aa"
    xs = [0.22, 0.37, 0.52, 0.67, 0.82]
    labels_a = ["A", "B", "C", "D", "E"]
    labels_b = ["A", "B", "C", "X", "Y"]

    ax = axes[0]
    panel(
        ax,
        "Jaccard : comparer deux trajets entiers",
        "On transforme chaque trajet en ensemble de cellules.",
        "#f8fafc",
    )
    ax.text(0.075, 0.62, "Trajet 1", ha="left", va="center", fontsize=10.5, weight="bold", color="#334155")
    ax.text(0.075, 0.47, "Trajet 2", ha="left", va="center", fontsize=10.5, weight="bold", color="#334155")
    path(ax, xs, 0.62, labels_a, [shared, shared, shared, only_a, only_a])
    path(ax, xs, 0.47, labels_b, [shared, shared, shared, only_b, only_b])
    ax.add_patch(Rectangle((0.18, 0.40), 0.42, 0.29, fill=False, edgecolor="#10b981", lw=2.4))
    ax.text(0.5, 0.31, "Cellules communes : A, B, C", ha="center", fontsize=12, weight="bold", color="#047857")
    ax.text(0.5, 0.23, "Score global = cellules communes / cellules totales", ha="center", fontsize=10.7, color="#0f172a")
    ax.text(0.5, 0.145, "Question : est-ce que les trajets complets se ressemblent assez ?", ha="center", fontsize=10.7, color="#b91c1c", weight="bold")

    ax = axes[1]
    panel(
        ax,
        "Segments ponderes : compter les morceaux communs",
        "On garde les transitions entre cellules voisines.",
        "#fffdf7",
    )
    ax.text(0.075, 0.62, "Trajet 1", ha="left", va="center", fontsize=10.5, weight="bold", color="#334155")
    ax.text(0.075, 0.47, "Trajet 2", ha="left", va="center", fontsize=10.5, weight="bold", color="#334155")
    path(ax, xs, 0.62, labels_a, [shared, shared, shared, only_a, only_a])
    path(ax, xs, 0.47, labels_b, [shared, shared, shared, only_b, only_b])
    ax.add_patch(Rectangle((0.18, 0.40), 0.34, 0.29, fill=False, edgecolor="#14b8a6", lw=2.4))
    ax.text(0.5, 0.31, "Transitions communes", ha="center", fontsize=12, weight="bold", color="#0f766e")
    ax.text(0.5, 0.225, "A -> B : poids 2    |    B -> C : poids 2", ha="center", fontsize=11.2, color="#0f172a")
    ax.text(0.5, 0.145, "Question : quels morceaux sont empruntes par plusieurs trajets ?", ha="center", fontsize=10.7, color="#b91c1c", weight="bold")

    fig.suptitle("Difference entre Jaccard et segments ponderes", fontsize=19, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Jaccard donne un score de ressemblance entre trajets complets. Les segments indiquent ou se trouvent les portions communes.",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    fig.tight_layout(rect=[0.02, 0.045, 0.98, 0.93])
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
