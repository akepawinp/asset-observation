"""Asset Observation - Historical financial data acquisition and analysis tools."""

from .downloader import (
    fetch_and_save_tickers,
    fetch_ticker_history,
    load_ticker_data,
    save_ticker_data,
    update_ticker_data,
)
from .metrics import (
    CANDIDATE_TICKERS,
    LOOKBACKS,
    MetricResult,
    compute_all,
    compute_metric_row,
)
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    generate_markdown_report,
    run_pipeline,
)
from .scoring import (
    DEFAULT_CONFIG,
    DeploymentLane,
    LaneRole,
    RegimeEvaluation,
    ScoringConfig,
    allocate_deployment_lanes,
    compute_aggregate_rankings,
    evaluate_candidate_models,
    evaluate_regime_stability,
    mean_reversion_quality,
    score_two_pillar,
)

__all__ = [
    # Downloader
    "fetch_ticker_history",
    "save_ticker_data",
    "update_ticker_data",
    "fetch_and_save_tickers",
    "load_ticker_data",
    # Metrics
    "CANDIDATE_TICKERS",
    "LOOKBACKS",
    "MetricResult",
    "compute_all",
    "compute_metric_row",
    # Scoring & Regimes
    "DEFAULT_CONFIG",
    "ScoringConfig",
    "DeploymentLane",
    "LaneRole",
    "RegimeEvaluation",
    "mean_reversion_quality",
    "score_two_pillar",
    "evaluate_candidate_models",
    "compute_aggregate_rankings",
    "evaluate_regime_stability",
    "allocate_deployment_lanes",
    # Pipeline Orchestrator
    "PipelineConfig",
    "PipelineResult",
    "generate_markdown_report",
    "run_pipeline",
]
