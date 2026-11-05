import pytest

from backend.app.pipeline.router import (
    RouterCapabilities,
    RoutingError,
    route_training,
)


def profile(*, rows: int, feature_count: int = 2, class_count: int = 2):
    columns = [
        {
            "name": f"feature_{index}",
            "dtype": "float64",
            "missing_count": 0,
            "missing_percent": 0,
            "cardinality": rows,
        }
        for index in range(feature_count)
    ]
    columns.append(
        {
            "name": "target",
            "dtype": "object",
            "missing_count": 0,
            "missing_percent": 0,
            "cardinality": class_count,
        }
    )
    return {"rows": rows, "columns": columns, "columns_count": len(columns)}


def route(
    *,
    rows: int,
    capabilities: RouterCapabilities,
    priority: str = "accuracy",
    feature_count: int = 2,
    path_override: str | None = None,
    hardware_constraint: str = "auto",
    performance_evidence: dict | None = None,
):
    return route_training(
        profile=profile(rows=rows, feature_count=feature_count),
        target_column="target",
        task_type="classification",
        test_size=0.2,
        priority=priority,
        hardware_constraint=hardware_constraint,
        path_override=path_override,
        capabilities=capabilities,
        performance_evidence=performance_evidence,
    )


def test_accuracy_defaults_to_gbdt_without_measured_performance_evidence() -> None:
    decision = route(
        rows=1_000,
        capabilities=RouterCapabilities(True, True, False),
        priority="accuracy",
    )

    assert decision.path == "gbdt"
    assert decision.engine == "autogluon.tabular"
    assert decision.device == "cpu"
    assert decision.training_rows == 800
    assert decision.feature_count == 2
    assert "no validated" in decision.rationale
    assert not decision.performance_evidence_used
    assert any("GPU unavailable" in reason for reason in decision.exclusions)


def test_speed_priority_uses_validated_scenario_matched_evidence() -> None:
    evidence = {
        "validated": True, "objective": "end_to_end_latency",
        "training_rows_min": 0, "training_rows_max": 10_000,
        "feature_count_min": 0, "feature_count_max": 10,
        "hardware": "cuda", "path": "tfm", "engine": "tabpfn",
        "p95_seconds": 2.5,
    }
    decision = route(
        rows=4_000,
        capabilities=RouterCapabilities(True, True, True, None, True, True),
        priority="speed",
        performance_evidence=evidence,
    )
    assert decision.path == "tfm"
    assert decision.engine == "tabpfn"
    assert decision.device == "cuda"
    assert decision.performance_evidence_used


def test_speed_priority_does_not_use_unvalidated_evidence() -> None:
    evidence = {
        "validated": False, "objective": "end_to_end_latency",
        "training_rows_min": 0, "training_rows_max": 10_000,
        "feature_count_min": 0, "feature_count_max": 10,
        "hardware": "cuda", "path": "tfm", "engine": "tabpfn",
        "p95_seconds": 2.5,
    }
    decision = route(
        rows=4_000,
        capabilities=RouterCapabilities(True, True, True, None, True, True),
        priority="speed",
        performance_evidence=evidence,
    )
    assert decision.path == "gbdt"
    assert not decision.performance_evidence_used


def test_cpu_only_constraint_uses_cpu_limits_even_when_gpu_is_available() -> None:
    decision = route(
        rows=2_000,
        capabilities=RouterCapabilities(True, True, True),
        priority="accuracy",
        hardware_constraint="cpu_only",
    )

    assert decision.path == "gbdt"
    assert decision.device == "cpu"
    assert any(
        "CPU-only execution was requested" in item for item in decision.exclusions
    )


def test_large_class_count_excludes_tabpfn_but_keeps_tabicl_candidate() -> None:
    decision = route_training(
        profile=profile(rows=4_000, class_count=161),
        target_column="target",
        task_type="classification",
        test_size=0.2,
        priority="accuracy",
        capabilities=RouterCapabilities(True, True, True, None, True, True),
    )

    assert decision.engine == "autogluon.tabular"
    assert any("TabPFN excluded" in item for item in decision.exclusions)


def test_cpu_dataset_above_conservative_tfm_limit_falls_back_to_autogluon() -> None:
    decision = route(
        rows=2_000,
        capabilities=RouterCapabilities(True, True, False),
        priority="accuracy",
    )

    assert decision.path == "gbdt"
    assert decision.engine == "autogluon.tabular"
    assert any("conservative CPU" in reason for reason in decision.exclusions)
    assert "no validated" in decision.rationale


def test_explainability_priority_selects_gbdt_and_records_tfm_exclusion() -> None:
    decision = route(
        rows=500,
        capabilities=RouterCapabilities(True, True, False, None, True, True),
        priority="explainability",
    )

    assert decision.path == "gbdt"
    assert decision.engine == "autogluon.tabular"
    assert any("MATLAB" in reason for reason in decision.exclusions)


def test_tfm_override_requires_an_eligible_installed_model() -> None:
        with pytest.raises(RoutingError, match="no ready TFM is eligible"):
        route(
            rows=500,
            capabilities=RouterCapabilities(False, False, False),
            path_override="tfm",
        )


def test_tfm_override_cannot_force_a_model_beyond_its_routing_limits() -> None:
        with pytest.raises(RoutingError, match="no ready TFM is eligible"):
        route(
            rows=10_000,
            capabilities=RouterCapabilities(True, True, False),
            path_override="tfm",
        )


def test_gbdt_override_is_persistable_even_when_tfm_is_eligible() -> None:
    decision = route(
        rows=500,
        capabilities=RouterCapabilities(True, True, False, None, True, True),
        path_override="gbdt",
    )

    assert decision.path == "gbdt"
    assert decision.engine == "autogluon.tabular"
    assert decision.override == "gbdt"
    assert "user explicitly selected" in decision.rationale


def test_large_feature_count_falls_back_when_tfms_exceed_supported_ranges() -> None:
    decision = route(
        rows=2_000,
        capabilities=RouterCapabilities(True, True, True),
        feature_count=6_001,
    )

    assert decision.path == "gbdt"
    assert any("features exceed" in reason for reason in decision.exclusions)


def test_package_presence_is_not_checkpoint_readiness() -> None:
    decision = route(
        rows=500,
        capabilities=RouterCapabilities(True, True, False),
    )
    assert decision.path == "gbdt"
    assert any("readiness has not been verified" in reason for reason in decision.exclusions)


def test_tfm_override_selects_only_a_ready_eligible_tfm() -> None:
    decision = route(
        rows=500,
        capabilities=RouterCapabilities(True, True, False, None, True, True),
        path_override="tfm",
    )
    assert decision.path == "tfm"
    assert decision.engine == "tabpfn"
    assert decision.selection_basis == "explicit_user_override"
