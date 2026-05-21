from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT = BASE_DIR / "results" / "focused_clustering_output" / "focused_profile_summary.csv"
OUTPUT = BASE_DIR / "results" / "report_assets" / "visualisation_profils_utilisateurs.png"

CRITERION_TITLES = {
    "AGE_GROUP": "Tranches d'age",
    "DOMINANT_MODE": "Mode dominant",
    "DRIVING_LICENCE": "Permis de conduire",
    "NAVIGO_SUB": "Abonnement Navigo",
    "NB_CAR_GROUP": "Voitures dans le foyer",
    "SEX": "Sexe",
}

COLORS = {
    "AGE_GROUP": "#2563eb",
    "DOMINANT_MODE": "#16a34a",
    "DRIVING_LICENCE": "#dc2626",
    "NAVIGO_SUB": "#9333ea",
    "NB_CAR_GROUP": "#ea580c",
    "SEX": "#0891b2",
}


def annotate_bars(axis):
    for patch in axis.patches:
        value = int(patch.get_width())
        axis.text(
            patch.get_width() + 1,
            patch.get_y() + patch.get_height() / 2,
            str(value),
            va="center",
            ha="left",
            fontsize=9,
            color="#334155",
        )


def plot_panel(axis, rows, criterion):
    rows = rows.sort_values("N_USERS", ascending=True)
    axis.barh(rows["VALUE"], rows["N_USERS"], color=COLORS.get(criterion, "#475569"))
    axis.set_title(CRITERION_TITLES.get(criterion, criterion), loc="left", fontsize=12, fontweight="bold")
    axis.set_xlabel("Nombre d'utilisateurs", fontsize=9)
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#cbd5e1")
    axis.tick_params(axis="y", labelsize=9, length=0)
    axis.tick_params(axis="x", labelsize=8, colors="#64748b")
    annotate_bars(axis)
    axis.set_xlim(0, max(rows["N_USERS"].max() * 1.22, 10))


def main():
    df = pd.read_csv(INPUT)
    criteria = [
        "SEX",
        "AGE_GROUP",
        "DOMINANT_MODE",
        "NAVIGO_SUB",
        "DRIVING_LICENCE",
        "NB_CAR_GROUP",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12), dpi=180)
    fig.patch.set_facecolor("#f8fafc")

    for axis, criterion in zip(axes.flatten(), criteria):
        rows = df[df["CRITERION"] == criterion]
        plot_panel(axis, rows, criterion)

    fig.suptitle(
        "Profils utilisateurs dans l'echantillon focalise",
        x=0.05,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.05,
        0.952,
        "Lecture exploratoire : les trajectoires sont regroupees selon les caracteristiques disponibles des individus.",
        ha="left",
        fontsize=10,
        color="#475569",
    )

    for axis in axes.flatten():
        axis.set_facecolor("#f8fafc")

    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.93], h_pad=2.3, w_pad=2.4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
