from pathlib import Path

import pandas as pd
import pytest

from backend.app.pipeline import gbdt_path
from backend.app.pipeline.gbdt_path import TrainingError, train_autogluon
from backend.app.pipeline.preprocessing import prepare_dataset


def make_classification_dataset(class_count: int = 2):
    labels = [f"class-{index % class_count}" for index in range(60)]
    return prepare_dataset(
        pd.DataFrame(
            {
                "number": list(range(60)),
                "category": ["a", "b"] * 30,
                "target": labels,
            }
        ),
        target_column="target",
        task_type="classification",
        test_size=0.2,
        random_state=42,
    )


def test_autogluon_adapter_selects_binary_cpu_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictor:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs
            Path(kwargs["path"]).mkdir(parents=True)

        def fit(self, **kwargs) -> None:
            captured["fit"] = kwargs

        def leaderboard(self, **kwargs) -> pd.DataFrame:
            captured["leaderboard"] = kwargs
            return pd.DataFrame(
                [{"model": "MockModel", "score_test": 0.9, "score_val": float("nan")}]
            )

    monkeypatch.setattr(gbdt_path, "TabularPredictor", FakePredictor)
    result = train_autogluon(
        prepared=make_classification_dataset(),
        target_column="target",
        task_type="classification",
        output_directory=tmp_path,
        time_limit_seconds=5,
    )

    assert captured["init"]["problem_type"] == "binary"
    assert captured["fit"]["num_gpus"] == 0
    assert captured["fit"]["time_limit"] == 5
    assert captured["fit"]["use_bag_holdout"] is True
    assert result.best_model == "MockModel"
    assert result.leaderboard[0]["score_val"] is None
    assert result.model_path.is_dir()
    assert result.leaderboard_path.is_file()


def test_multiclass_rejects_unsupported_metric_before_autogluon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def predictor_must_not_be_created(**kwargs) -> None:
        raise AssertionError(f"Predictor initialized unexpectedly: {kwargs}")

    monkeypatch.setattr(gbdt_path, "TabularPredictor", predictor_must_not_be_created)
    with pytest.raises(TrainingError, match="not supported for multiclass"):
        train_autogluon(
            prepared=make_classification_dataset(class_count=3),
            target_column="target",
            task_type="classification",
            output_directory=tmp_path,
            time_limit_seconds=5,
            evaluation_metric="roc_auc",
        )
