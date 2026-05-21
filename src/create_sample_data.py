from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
FOCUSED_POINTS = BASE / "results" / "focused_clustering_output" / "focused_50_refined_trajectories.csv"
SELECTED_USERS = BASE / "results" / "focused_clustering_output" / "selected_same_zone_users.csv"
INDIVIDUALS = BASE / "NetMob25CleanedData" / "NetMob25CleanedData" / "individuals_dataset.csv"
DISPLACEMENTS = BASE / "NetMob25CleanedData" / "NetMob25CleanedData" / "displacements_dataset.csv"
OUT = BASE / "sample_data"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    selected_users = pd.read_csv(SELECTED_USERS)
    selected_users["USER_ID"] = selected_users["USER_ID"].astype(str)
    selected_user_ids = set(selected_users["USER_ID"])

    points = pd.read_csv(FOCUSED_POINTS)
    points["USER_ID"] = points["USER_ID"].astype(str)
    points = points.sort_values(["TRAJECTORY_UID", "POINT_INDEX"])

    # Keep the demo dataset light while preserving the shape of trajectories.
    sampled = points.groupby("TRAJECTORY_UID", group_keys=False).apply(
        lambda group: group.iloc[:: max(1, len(group) // 80)]
    )
    if "TRAJECTORY_UID" not in sampled.columns:
        sampled = sampled.reset_index()
    if "TRAJECTORY_UID" not in sampled.columns:
        sampled["TRAJECTORY_UID"] = (
            sampled["USER_ID"].astype(str) + "_" + sampled["TRAJECTORY_ID"].astype(str)
        )
    sampled = sampled[[column for column in points.columns if column in sampled.columns]]
    sampled.to_csv(OUT / "focused_demo_refined_trajectories.csv", index=False)
    selected_users.to_csv(OUT / "selected_same_zone_users.csv", index=False)

    individuals = pd.read_csv(INDIVIDUALS)
    individuals["ID"] = individuals["ID"].astype(str)
    individuals[individuals["ID"].isin(selected_user_ids)].to_csv(
        OUT / "individuals_dataset.csv",
        index=False,
    )

    displacements = pd.read_csv(DISPLACEMENTS)
    displacements["ID"] = displacements["ID"].astype(str)
    displacements[displacements["ID"].isin(selected_user_ids)].to_csv(
        OUT / "displacements_dataset.csv",
        index=False,
    )

    print(f"Demo points: {len(sampled):,}")
    print(f"Demo users : {sampled['USER_ID'].nunique():,}")
    print(OUT)


if __name__ == "__main__":
    main()
