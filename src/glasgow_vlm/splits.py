from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def stratified_group_split(
    records: list[dict],
    group_key: str = "datazone",
    label_key: str = "deprivation_quintile",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    groups: dict[object, list[dict]] = defaultdict(list)
    group_labels: dict[object, object] = {}
    for record in records:
        group = record.get(group_key)
        groups[group].append(record)
        group_labels[group] = record.get(label_key)

    by_label: dict[object, list[object]] = defaultdict(list)
    for group, label in group_labels.items():
        by_label[label].append(group)

    rng = random.Random(seed)
    train_groups = set()
    val_groups = set()
    test_groups = set()

    for label, label_groups in by_label.items():
        rng.shuffle(label_groups)
        n = len(label_groups)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        n_test = max(0, n - n_train - n_val)

        train_groups.update(label_groups[:n_train])
        val_groups.update(label_groups[n_train : n_train + n_val])
        test_groups.update(label_groups[n_train + n_val : n_train + n_val + n_test])

    def collect(selected_groups: set[object]) -> list[dict]:
        output: list[dict] = []
        for group in selected_groups:
            output.extend(groups[group])
        return output

    return collect(train_groups), collect(val_groups), collect(test_groups)


DEFAULT_SHP = "dataset/glasgow_datazone/glasgow_datazone.shp"
DEFAULT_SIMD = "dataset/SIMD/SIMD_precise.csv"


def load_datazone_iz_map(shp_path: str | Path = DEFAULT_SHP) -> pd.DataFrame:
    gdf = gpd.read_file(shp_path)
    return gdf[["DataZone", "Intermedia"]].rename(
        columns={"DataZone": "datazone", "Intermedia": "intermediate_zone"}
    )


def group_kfold_by_iz(
    datazones: list[str],
    n_splits: int = 5,
    strata_col: str = "Overall",
    n_strata: int = 5,
    shp_path: str | Path = DEFAULT_SHP,
    simd_path: str | Path = DEFAULT_SIMD,
    seed: int = 42,
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (train_datazones, val_datazones) for each of K folds.

    Groups are Intermediate Zones (cuts spatial autocorrelation). Folds are
    stratified on quantile bins of `strata_col` from SIMD_precise.csv so each
    fold has a comparable SIMD distribution.
    """
    iz_map = load_datazone_iz_map(shp_path)
    simd = pd.read_csv(simd_path)

    df = pd.DataFrame({"datazone": list(datazones)}).merge(iz_map, on="datazone")
    df = df.merge(simd[["datazone", strata_col]], on="datazone")
    df["strata"] = pd.qcut(df[strata_col], q=n_strata, labels=False, duplicates="drop")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dz = df["datazone"].to_numpy()
    for tr_idx, va_idx in cv.split(df, df["strata"], groups=df["intermediate_zone"]):
        yield dz[tr_idx].tolist(), dz[va_idx].tolist()
