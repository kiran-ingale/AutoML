from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.schemas.preprocessing import PreprocessingSummary

TaskType = Literal["classification", "regression"]


class PreprocessingError(ValueError):
    """Raised when a dataset cannot be prepared for model training."""


@dataclass(frozen=True)
class PreparedDataset:
    X_train_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    X_train: pd.DataFrame | sparse.spmatrix
    X_validation: pd.DataFrame | sparse.spmatrix
    y_train: pd.Series
    y_validation: pd.Series
    transformer: ColumnTransformer
    summary: PreprocessingSummary


def prepare_dataset(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    task_type: TaskType,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedDataset:
    if dataframe.empty:
        raise PreprocessingError("Dataset must contain at least one row.")
    if not 0 < test_size < 1:
        raise PreprocessingError("test_size must be greater than 0 and less than 1.")
    if task_type not in ("classification", "regression"):
        raise PreprocessingError("task_type must be classification or regression.")

    dataframe = dataframe.copy()
    dataframe.columns = dataframe.columns.map(str)
    if target_column not in dataframe.columns:
        raise PreprocessingError(f"Target column {target_column!r} was not found.")
    if dataframe.columns.has_duplicates:
        raise PreprocessingError("Dataset column names must be unique.")

    rows_before_target_filter = len(dataframe)
    usable = dataframe.loc[dataframe[target_column].notna()].copy()
    rows_without_target = rows_before_target_filter - len(usable)
    if len(usable) < 2:
        raise PreprocessingError(
            "At least two rows with a non-missing target are required."
        )

    y = usable[target_column].copy()
    X = usable.drop(columns=[target_column]).copy()
    if X.empty:
        raise PreprocessingError(
            "At least one feature column besides the target is required."
        )
    if task_type == "regression" and not is_numeric_dtype(y):
        raise PreprocessingError("Regression target values must be numeric.")
    if task_type == "regression" and not np.isfinite(y.to_numpy(dtype=float)).all():
        raise PreprocessingError("Regression target values must be finite.")

    if y.nunique(dropna=True) < 2:
        raise PreprocessingError("Target must contain at least two distinct values.")

    numeric_columns = [
        str(column)
        for column in X.columns
        if is_numeric_dtype(X[column]) and not pd.api.types.is_bool_dtype(X[column])
    ]
    categorical_columns = [
        str(column) for column in X.columns if column not in numeric_columns
    ]
    missing_feature_values = int(X.isna().sum().sum())
    if numeric_columns:
        X[numeric_columns] = X[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    stratify = None
    split_method = "random"
    if task_type == "classification":
        class_counts = y.value_counts(dropna=True)
        class_count = len(class_counts)
        validation_rows = int(np.ceil(len(y) * test_size))
        training_rows = len(y) - validation_rows
        if (
            int(class_counts.min()) < 2
            or validation_rows < class_count
            or training_rows < class_count
        ):
            raise PreprocessingError(
                "Classification split cannot preserve every class in both "
                "training and validation sets. Add more examples per class "
                "or choose a larger dataset."
            )
        stratify = y
        split_method = "stratified"

    try:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise PreprocessingError(f"Could not split dataset: {exc}") from exc

    constant_columns = [
        str(column)
        for column in X_train.columns
        if X_train[column].nunique(dropna=False) <= 1
    ]
    X_train = X_train.drop(columns=constant_columns)
    X_validation = X_validation.drop(columns=constant_columns)
    numeric_columns = [c for c in numeric_columns if c not in constant_columns]
    categorical_columns = [c for c in categorical_columns if c not in constant_columns]

    if not numeric_columns and not categorical_columns:
        raise PreprocessingError(
            "No usable feature columns remain after removing constant columns."
        )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__automl_missing__",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                max_categories=100,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    try:
        X_train_transformed = transformer.fit_transform(X_train)
        X_validation_transformed = transformer.transform(X_validation)
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(f"Feature preprocessing failed: {exc}") from exc

    if sparse.issparse(X_train_transformed):
        output_feature_count = int(X_train_transformed.shape[1])
        X_train_result: pd.DataFrame | sparse.spmatrix = X_train_transformed
        X_validation_result: pd.DataFrame | sparse.spmatrix = X_validation_transformed
    else:
        feature_names = transformer.get_feature_names_out()
        output_feature_count = len(feature_names)
        X_train_result = pd.DataFrame(
            X_train_transformed,
            columns=feature_names,
            index=X_train.index,
        )
        X_validation_result = pd.DataFrame(
            X_validation_transformed,
            columns=feature_names,
            index=X_validation.index,
        )

    warnings: list[str] = []
    if rows_without_target:
        warnings.append(
            f"Dropped {rows_without_target} row(s) with a missing target value."
        )
    if constant_columns:
        warnings.append(
            "Dropped constant feature column(s): " + ", ".join(constant_columns) + "."
        )
    if missing_feature_values:
        warnings.append(
            f"Imputed {missing_feature_values} missing feature value(s) using "
            "training-set statistics."
        )
    if numeric_columns:
        warnings.append(
            "Numeric features were median-imputed and standardized using "
            "training-set statistics."
        )
    if categorical_columns:
        warnings.append(
            "Categorical features were missing-value-filled and one-hot encoded; "
            "unseen validation categories are ignored."
        )

    summary = PreprocessingSummary(
        target_column=target_column,
        task_type=task_type,
        split_method=split_method,
        test_size=test_size,
        random_state=random_state,
        original_rows=rows_before_target_filter,
        rows_without_target=rows_without_target,
        training_rows=len(y_train),
        validation_rows=len(y_validation),
        input_feature_count=len(dataframe.columns) - 1,
        output_feature_count=output_feature_count,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        dropped_constant_columns=constant_columns,
        missing_feature_values_before_imputation=missing_feature_values,
        warnings=warnings,
    )

    return PreparedDataset(
        X_train_raw=X_train,
        X_validation_raw=X_validation,
        X_train=X_train_result,
        X_validation=X_validation_result,
        y_train=y_train,
        y_validation=y_validation,
        transformer=transformer,
        summary=summary,
    )
