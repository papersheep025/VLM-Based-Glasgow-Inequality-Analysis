from __future__ import annotations

import random
from collections import defaultdict


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

