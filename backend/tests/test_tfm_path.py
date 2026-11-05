from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from backend.app.pipeline import tfm_path
from backend.app.pipeline.preprocessing import prepare_dataset


class PersistableDummyClassifier(DummyClassifier):
    def save_fit_state(self, path: Path) -> None:
        path.write_text("mock fitted state", encoding="utf-8")


class PersistableDummyRegressor(DummyRegressor):
    def save(self, path: Path, **kwargs) -> None:
        del kwargs
        path.write_text("mock fitted state", encoding="utf-8")


def test_tfm_training_persists_model_and_validation_leaderboard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataframe = pd.DataFrame(
        {
            "number": range(40),
            "category": ["a", "b"] * 20,
            "target": ["no", "yes"] * 20,
        }
    )
    prepared = prepare_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
        test_size=0.2,
        random_state=42,
    )
    monkeypatch.setattr(
        tfm_path,
        "_create_tfm",
        lambda **kwargs: PersistableDummyClassifier(strategy="most_frequent"),
    )

    result = tfm_path.train_tfm(
        engine="tabpfn",
        prepared=prepared,
        task_type="classification",
        output_directory=tmp_path,
        evaluation_metric="accuracy",
    )

    assert result.best_model == "PersistableDummyClassifier"
    assert result.leaderboard[0]["eval_metric"] == "accuracy"
    assert result.leaderboard[0]["score_test"] == 0.5
    assert (result.model_path / "predictor.tabpfn_fit").is_file()
    assert result.leaderboard_path.is_file()


def test_tfm_metric_rejects_roc_auc_for_multiclass(tmp_path: Path, monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {
            "number": range(60),
            "target": [f"class-{index % 3}" for index in range(60)],
        }
    )
    prepared = prepare_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
        test_size=0.2,
        random_state=42,
    )
    monkeypatch.setattr(
        tfm_path,
        "_create_tfm",
        lambda **kwargs: PersistableDummyClassifier(strategy="most_frequent"),
    )

    with pytest.raises(tfm_path.TrainingError, match="not supported for multiclass"):
        tfm_path.train_tfm(
            engine="tabicl",
            prepared=prepared,
            task_type="classification",
            output_directory=tmp_path,
            evaluation_metric="roc_auc",
        )


def test_tfm_regression_returns_higher_is_better_rmse_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataframe = pd.DataFrame(
        {"number": range(40), "target": [float(index) for index in range(40)]}
    )
    prepared = prepare_dataset(
        dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.2,
        random_state=42,
    )
    monkeypatch.setattr(
        tfm_path,
        "_create_tfm",
        lambda **kwargs: PersistableDummyRegressor(strategy="mean"),
    )

    result = tfm_path.train_tfm(
        engine="tabicl",
        prepared=prepared,
        task_type="regression",
        output_directory=tmp_path,
        evaluation_metric="root_mean_squared_error",
    )

    assert result.best_model == "PersistableDummyRegressor"
    assert result.leaderboard[0]["score_test"] < 0
    assert (result.model_path / "predictor.pkl").is_file()
