import importlib.util
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

TaskType = Literal["classification", "regression"]
ModelPath = Literal["gbdt", "tfm"]
Priority = Literal["accuracy", "speed", "explainability"]
HardwareConstraint = Literal["auto", "cpu_only"]

TABPFN_CPU_ROW_LIMIT = 1_000
TABPFN_GPU_ROW_LIMIT = 100_000
TABPFN_RECOMMENDED_FEATURE_LIMIT = 6_000
TABPFN_CLASS_LIMIT = 160
TABICL_GPU_ROW_LIMIT = 100_000
TABICL_FEATURE_LIMIT = 2_000
TFM_CPU_ROW_LIMIT = 1_000

# Conservative application policies, not vendor hard limits. Package presence
# is only a coarse capability signal; checkpoint/license readiness is unknown
# until the model is initialized and the fit begins.


class RoutingError(ValueError):
    """Raised when a requested route cannot be selected."""


@dataclass(frozen=True)
class RouterCapabilities:
    tabpfn_available: bool
    tabicl_available: bool
    gpu_available: bool
    gpu_detection_error: str | None = None
    # Set true only when the deployment has explicitly verified access to the
    # required checkpoint and license. False does not reveal secret values.
    tabpfn_ready: bool = False
    tabicl_ready: bool = False


@dataclass(frozen=True)
class RouteDecision:
    path: ModelPath
    engine: str
    priority: Priority
    hardware_constraint: HardwareConstraint
    device: Literal["cpu", "cuda"]
    training_rows: int
    feature_count: int
    target_class_count: int | None
    rationale: str
    exclusions: list[str]
    override: ModelPath | None
    selection_basis: str = "deterministic_requirements_and_eligibility_policy"
    performance_evidence_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_router_capabilities() -> RouterCapabilities:
    tabpfn_available = importlib.util.find_spec("tabpfn") is not None
    tabicl_available = importlib.util.find_spec("tabicl") is not None
    gpu_available = False
    gpu_detection_error = None
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except (ImportError, OSError, RuntimeError) as exc:
            gpu_detection_error = type(exc).__name__
    # Only inspect whether the secret is present; never read it into a route,
    # log line, exception, or status payload. License acceptance is an explicit
    # deployment attestation because package APIs do not provide a safe,
    # offline, authoritative check for the user's acceptance state.
    tabpfn_token_configured = bool(os.environ.get("TABPFN_TOKEN", "").strip())
    tabpfn_license_accepted = (
        os.environ.get("TABPFN_LICENSE_ACCEPTED", "").casefold() == "true"
    )
    tabicl_ready = os.environ.get("TABICL_READY", "").casefold() == "true"
    return RouterCapabilities(
        tabpfn_available=tabpfn_available,
        tabicl_available=tabicl_available,
        gpu_available=gpu_available,
        gpu_detection_error=gpu_detection_error,
        # A TabPFN token alone is insufficient: the operator must also attest
        # that the current checkpoint terms were accepted. TabICL readiness is
        # an operator attestation that the pinned checkpoint can be accessed.
        tabpfn_ready=(
            tabpfn_available
            and tabpfn_token_configured
            and tabpfn_license_accepted
        ),
        tabicl_ready=tabicl_available and tabicl_ready,
    )


def _profile_values(
    profile: dict[str, Any],
    *,
    target_column: str,
    task_type: TaskType,
    test_size: float,
) -> tuple[int, int, int | None]:
    try:
        rows = int(profile["rows"])
        columns = profile["columns"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RoutingError("Dataset profile is incomplete for model routing.") from exc

    if not isinstance(columns, list):
        raise RoutingError("Dataset profile columns are invalid.")
    target_profile = next(
        (
            column
            for column in columns
            if isinstance(column, dict) and column.get("name") == target_column
        ),
        None,
    )
    if target_profile is None:
        raise RoutingError(
            f"Target column {target_column!r} is missing from the dataset profile."
        )

    feature_columns = [
        column
        for column in columns
        if isinstance(column, dict) and column.get("name") != target_column
    ]
    training_rows = rows - math.ceil(rows * test_size)
    class_count = (
        int(target_profile.get("cardinality", 0))
        if task_type == "classification"
        else None
    )
    return training_rows, len(feature_columns), class_count


def route_training(
    *,
    profile: dict[str, Any],
    target_column: str,
    task_type: TaskType,
    test_size: float,
    priority: Priority = "accuracy",
    hardware_constraint: HardwareConstraint = "auto",
    path_override: ModelPath | None = None,
    capabilities: RouterCapabilities | None = None,
    performance_evidence: dict[str, Any] | None = None,
) -> RouteDecision:
    capabilities = capabilities or detect_router_capabilities()
    training_rows, feature_count, class_count = _profile_values(
        profile,
        target_column=target_column,
        task_type=task_type,
        test_size=test_size,
    )
    if not feature_count:
        raise RoutingError("At least one feature column is required for routing.")
    if task_type == "classification" and (class_count is None or class_count < 2):
        raise RoutingError(
            "Classification routing requires at least two target classes."
        )

    use_gpu = capabilities.gpu_available and hardware_constraint != "cpu_only"
    device: Literal["cpu", "cuda"] = "cuda" if use_gpu else "cpu"
    exclusions: list[str] = []
    eligible: list[str] = []

    if not capabilities.tabpfn_available:
        exclusions.append("TabPFN excluded: the tabpfn package is not installed.")
    elif not capabilities.tabpfn_ready:
        exclusions.append(
            "TabPFN excluded: checkpoint/license readiness has not been verified."
        )
    else:
        row_limit = TABPFN_GPU_ROW_LIMIT if use_gpu else TABPFN_CPU_ROW_LIMIT
        if training_rows > row_limit:
            exclusions.append(
                f"TabPFN excluded: {training_rows:,} training rows exceed the "
                f"{'GPU' if use_gpu else 'conservative CPU'} routing limit of "
                f"{row_limit:,}."
            )
        elif feature_count > TABPFN_RECOMMENDED_FEATURE_LIMIT:
            exclusions.append(
                f"TabPFN excluded: {feature_count:,} features exceed the "
                f"recommended limit of {TABPFN_RECOMMENDED_FEATURE_LIMIT:,}."
            )
        elif (
            task_type == "classification"
            and class_count is not None
            and class_count > TABPFN_CLASS_LIMIT
        ):
            exclusions.append(
                f"TabPFN excluded: {class_count:,} target classes exceed its "
                f"{TABPFN_CLASS_LIMIT}-class native limit."
            )
        else:
            eligible.append("tabpfn")

    if not capabilities.tabicl_available:
        exclusions.append("TabICL excluded: the tabicl package is not installed.")
    elif not capabilities.tabicl_ready:
        exclusions.append(
            "TabICL excluded: checkpoint/license readiness has not been verified."
        )
    else:
        row_limit = TABICL_GPU_ROW_LIMIT if use_gpu else TFM_CPU_ROW_LIMIT
        if training_rows < 300:
            exclusions.append(
                f"TabICL excluded: {training_rows:,} training rows are below the "
                "300-row range used in its published benchmark guidance."
            )
        elif training_rows > row_limit:
            exclusions.append(
                f"TabICL excluded: {training_rows:,} training rows exceed the "
                f"{'GPU benchmark' if use_gpu else 'conservative CPU'} routing "
                f"limit of {row_limit:,}."
            )
        elif feature_count > TABICL_FEATURE_LIMIT:
            exclusions.append(
                f"TabICL excluded: {feature_count:,} features exceed its "
                f"{TABICL_FEATURE_LIMIT:,}-feature benchmark range."
            )
        else:
            eligible.append("tabicl")

    # Priority determines policy intent, but cannot override hard capability
    # gates or fabricate evidence that one model is faster/more accurate.
    if path_override == "gbdt":
        selected_path: ModelPath = "gbdt"
        engine = "autogluon.tabular"
        rationale = "The user explicitly selected the AutoGluon/GBDT path."
    elif path_override == "tfm":
        if not eligible:
            raise RoutingError(
                "The TFM path was requested, but no ready TFM is eligible: "
                + " ".join(exclusions)
            )
        selected_path = "tfm"
        engine = _select_tfm(eligible, training_rows=training_rows)
        rationale = _tfm_rationale(
            engine=engine,
            training_rows=training_rows,
            feature_count=feature_count,
            device=device,
            priority=priority,
            explicit=True,
        )
    elif priority == "explainability":
        selected_path = "gbdt"
        engine = "autogluon.tabular"
        exclusions.append(
            "TFM path not selected: project explainability requires MATLAB, "
            "and MATLAB readiness is not an input to this router."
        )
        rationale = (
            "AutoGluon/GBDT was selected because explainability is the stated "
            "priority and the planned TreeSHAP path is available in the product "
            "design; TFM MATLAB readiness is not verified here."
        )
    elif priority == "speed":
        selected_path, engine, rationale = _choose_by_speed_evidence(
            eligible=eligible,
            performance_evidence=performance_evidence,
            exclusions=exclusions,
            training_rows=training_rows,
            feature_count=feature_count,
        )
    else:
        # Accuracy preference alone is not evidence of a family advantage.
        # Choose the mature GBDT baseline unless a validated, applicable
        # performance policy is supplied or the user explicitly requests TFMs.
        selected_path, engine = "gbdt", "autogluon.tabular"
        rationale = (
            "AutoGluon/GBDT was selected: accuracy is the priority, but no "
            "validated, applicable cross-family performance evidence is "
            "available to justify choosing a TFM."
        )
        if eligible:
            exclusions.append(
                "Eligible TFMs were not selected because eligibility alone does "
                "not establish higher accuracy."
            )

    if capabilities.gpu_detection_error:
        exclusions.append(
            "GPU capability detection failed "
            f"({capabilities.gpu_detection_error}); CPU-only routing limits were applied."
        )
    elif not capabilities.gpu_available:
        exclusions.append("GPU unavailable; CPU-only routing limits were applied.")
    elif hardware_constraint == "cpu_only":
        exclusions.append("CPU-only execution was requested; GPU was not selected.")

    route_device: Literal["cpu", "cuda"] = device if selected_path == "tfm" else "cpu"
    return RouteDecision(
        path=selected_path,
        engine=engine,
        priority=priority,
        hardware_constraint=hardware_constraint,
        device=route_device,
        training_rows=training_rows,
        feature_count=feature_count,
        target_class_count=class_count,
        rationale=rationale,
        exclusions=exclusions,
        override=path_override,
        selection_basis=(
            "explicit_user_override"
            if path_override
            else "validated_speed_evidence"
            if priority == "speed" and "evidence supports" in rationale
            else "requirements_and_conservative_fallback"
        ),
        performance_evidence_used=(
            priority == "speed" and "evidence supports" in rationale
        ),
    )


def _choose_by_speed_evidence(
    *,
    eligible: list[str],
    performance_evidence: dict[str, Any] | None,
    exclusions: list[str],
    training_rows: int,
    feature_count: int,
) -> tuple[ModelPath, str, str]:
    """Use only prevalidated, scenario-matched end-to-end speed evidence."""
    if performance_evidence:
        evidence = performance_evidence
        if (
            evidence.get("validated") is True
            and evidence.get("objective") == "end_to_end_latency"
            and evidence.get("training_rows_min", 0) <= training_rows
            and evidence.get("training_rows_max", -1) >= training_rows
            and evidence.get("feature_count_min", 0) <= feature_count
            and evidence.get("feature_count_max", -1) >= feature_count
            and evidence.get("hardware") in {"cpu", "cuda"}
            and evidence.get("path") in {"gbdt", "tfm"}
            and (
                evidence["path"] == "gbdt"
                or evidence.get("engine") in eligible
            )
            and isinstance(evidence.get("p95_seconds"), (int, float))
            and evidence["p95_seconds"] > 0
        ):
            path: ModelPath = evidence["path"]
            engine = (
                "autogluon.tabular"
                if path == "gbdt"
                else str(evidence["engine"])
            )
            return (
                path,
                engine,
                f"{engine} was selected for speed because validated, scenario-matched "
                f"end-to-end evidence supports it (p95 {evidence['p95_seconds']:.3g}s).",
            )

    exclusions.append(
        "Speed priority has no validated, scenario-matched end-to-end latency "
        "evidence; no speed advantage is assumed from model family."
    )
    return (
        "gbdt",
        "autogluon.tabular",
        "AutoGluon/GBDT was selected because speed is the priority, but no "
        "validated end-to-end latency evidence supports a TFM route for this "
        "profile and hardware scenario.",
    )


def _select_tfm(eligible: list[str], *, training_rows: int) -> str:
    """Choose a TFM by documented scale policy, not claimed superiority."""
    if len(eligible) == 1:
        return eligible[0]
    if training_rows <= TABPFN_CPU_ROW_LIMIT and "tabpfn" in eligible:
        return "tabpfn"
    if "tabicl" in eligible:
        return "tabicl"
    return eligible[0]


def _tfm_rationale(
    *,
    engine: str,
    training_rows: int,
    feature_count: int,
    device: Literal["cpu", "cuda"],
    priority: Priority,
    explicit: bool,
) -> str:
    model_name = "TabPFN" if engine == "tabpfn" else "TabICL"
    device_limit = (
        (TABPFN_GPU_ROW_LIMIT if device == "cuda" else TABPFN_CPU_ROW_LIMIT)
        if engine == "tabpfn"
        else (TABICL_GPU_ROW_LIMIT if device == "cuda" else TFM_CPU_ROW_LIMIT)
    )
    origin = "The user explicitly requested the TFM path. " if explicit else ""
    return (
        f"{origin}{model_name} was selected within the eligible TFM set for the "
        f"{priority} priority: {training_rows:,} training rows and {feature_count:,} "
        f"features fit the conservative {device} policy ({device_limit:,} rows). "
        "This scale rule does not claim predictive or runtime superiority."
    )
