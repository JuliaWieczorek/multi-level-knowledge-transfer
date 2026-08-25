from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


ORIGINAL_4 = "original4"
COARSE_3 = "coarse3"
OUTCOME_LABEL_SCHEMES = (ORIGINAL_4, COARSE_3)


@dataclass(frozen=True)
class OutcomeLabelScheme:
    name: str
    labels: tuple[int, ...]
    label_names: tuple[str, ...]
    raw_groups: dict[int, tuple[int, ...]]

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def to_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "labels": list(self.labels),
            "label_names": list(self.label_names),
            "raw_groups": {
                str(label): list(values)
                for label, values in self.raw_groups.items()
            },
        }


_SCHEMES = {
    ORIGINAL_4: OutcomeLabelScheme(
        name=ORIGINAL_4,
        labels=(1, 2, 3, 4),
        label_names=("1", "2", "3", "4"),
        raw_groups={1: (1,), 2: (2,), 3: (3,), 4: (4,)},
    ),
    COARSE_3: OutcomeLabelScheme(
        name=COARSE_3,
        labels=(1, 2, 3),
        label_names=("low_1_2", "middle_3", "high_4_5"),
        raw_groups={1: (1, 2), 2: (3,), 3: (4, 5)},
    ),
}


def get_outcome_label_scheme(name: str | None = None) -> OutcomeLabelScheme:
    normalised = name or ORIGINAL_4
    try:
        return _SCHEMES[normalised]
    except KeyError as error:
        raise ValueError(
            f"Unsupported outcome label scheme {normalised!r}; expected one of "
            f"{OUTCOME_LABEL_SCHEMES}."
        ) from error


def map_outcome_values(
    values: Iterable[int], scheme_name: str | None = None
) -> np.ndarray:
    scheme = get_outcome_label_scheme(scheme_name)
    raw = np.asarray(list(values), dtype=int)
    if raw.size == 0:
        return raw
    lookup = {
        raw_value: label
        for label, raw_values in scheme.raw_groups.items()
        for raw_value in raw_values
    }
    unknown = sorted(set(raw.tolist()) - set(lookup))
    if unknown:
        raise ValueError(
            f"Values {unknown} cannot be mapped by outcome label scheme "
            f"{scheme.name!r}."
        )
    return np.asarray([lookup[int(value)] for value in raw], dtype=int)


def relabel_outcome_frame(
    frame: pd.DataFrame, scheme_name: str | None = None
) -> pd.DataFrame:
    """Return a copy with final/drop targets mapped under the selected scheme.

    Raw outcome columns are retained so predictions and diagnostics remain
    traceable to the original five-point survey scale.
    """
    required = {"final_intensity", "drop_magnitude"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Outcome frame is missing columns: {sorted(missing)}")
    scheme = get_outcome_label_scheme(scheme_name)
    result = frame.copy()
    for column in ("final_intensity", "drop_magnitude"):
        raw_column = f"{column}_raw"
        if raw_column not in result:
            result[raw_column] = result[column].astype(int)
        result[column] = map_outcome_values(result[raw_column], scheme.name)
    result["label_scheme"] = scheme.name
    return result
