import csv
from collections import Counter
from io import StringIO
from typing import Any

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when an uploaded file is not a usable CSV dataset."""


def profile_csv(contents: bytes) -> dict[str, Any]:
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(
            "CSV must use UTF-8 or UTF-8 with BOM encoding."
        ) from exc

    try:
        header = next(csv.reader(StringIO(text)))
    except StopIteration as exc:
        raise DatasetValidationError("CSV file is empty.") from exc
    except csv.Error as exc:
        raise DatasetValidationError("CSV header could not be parsed.") from exc

    if not header or all(not name.strip() for name in header):
        raise DatasetValidationError("CSV must contain at least one column header.")
    if any(not name.strip() for name in header):
        raise DatasetValidationError("CSV column headers cannot be blank.")

    duplicate_headers = sorted(
        name for name, count in Counter(header).items() if count > 1
    )
    if duplicate_headers:
        raise DatasetValidationError(
            "CSV contains duplicate column headers: "
            + ", ".join(repr(name) for name in duplicate_headers)
        )

    try:
        dataframe = pd.read_csv(StringIO(text), low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError("CSV file has no tabular data.") from exc
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise DatasetValidationError(
            "CSV could not be parsed; check its delimiter and row structure."
        ) from exc

    if dataframe.empty:
        raise DatasetValidationError("CSV must contain at least one data row.")
    if dataframe.shape[1] != len(header):
        raise DatasetValidationError("CSV rows do not match the header column count.")

    row_count, column_count = dataframe.shape
    columns: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name in dataframe.columns:
        series = dataframe[name]
        missing_count = int(series.isna().sum())
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_percent": round(missing_count / row_count * 100, 2),
                "cardinality": int(series.nunique(dropna=True)),
            }
        )
        if missing_count:
            warnings.append(
                f"Column {name!r} has {missing_count} missing value(s) "
                f"({missing_count / row_count * 100:.2f}%)."
            )

    duplicate_rows = int(dataframe.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"Dataset contains {duplicate_rows} duplicate row(s).")

    target_column_guess = str(dataframe.columns[-1])
    target = dataframe.iloc[:, -1].dropna()
    class_imbalance_ratio: float | None = None
    if 2 <= target.nunique() <= 20:
        class_counts = target.value_counts()
        smallest_class = int(class_counts.min())
        if smallest_class:
            class_imbalance_ratio = round(
                int(class_counts.max()) / smallest_class,
                2,
            )

    return {
        "rows": int(row_count),
        "columns_count": int(column_count),
        "columns": columns,
        "duplicate_rows": duplicate_rows,
        "target_column_guess": target_column_guess,
        "target_guess_method": "last_column_heuristic_requires_user_confirmation",
        "class_imbalance_ratio": class_imbalance_ratio,
        "warnings": warnings,
    }
