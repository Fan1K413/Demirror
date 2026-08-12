"""AI-assisted surface diagnostics isolated from the human continuation gate."""

from __future__ import annotations

import statistics
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.geometry_ai.deterministic_surfaces import (
    DeterministicSurfaceBaselineResult,
)
from image_trust.geometry_ai.measurement_types import GeometryMeasurementV2Result
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
)
from image_trust.geometry_ai.relation_validation import (
    validate_annotation_semantic_closure,
)
from image_trust.geometry_ai.surface_comparison import (
    DEFAULT_SURFACE_COMPARISON_CONFIG,
    BinaryDecisionCounts,
    FamilySurfaceComparison,
    SurfaceComparisonConfig,
    _MutableBinaryCounts,
    _at_least,
    _canonical_hash,
    _compare_family_pairs,
    _unassessable_family_record,
    _update_family_counts,
    _validate_identity_and_baseline,
)
from image_trust.geometry_ai.surface_conditioned import (
    SurfaceConditionedReplayResult,
    assess_surface_conditioned_g1_g4,
)


AGENT_COMPARISON_SCHEMA_VERSION = "geometry-surface-agent-comparison-v1"
AGENT_AUTHORIZATION_SCHEMA_VERSION = (
    "geometry-surface-agent-diagnostic-replay-authorization-v1"
)
AGENT_REPLAY_SCHEMA_VERSION = "geometry-surface-agent-conditioned-g1-g4-v1"
ANNOTATION_SEMANTICS = "AI-assisted source-blind preannotation; not human ground truth"


class _FrozenAgentDiagnosticContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AgentSurfaceComparisonReport(_FrozenAgentDiagnosticContract):
    schema_version: Literal[AGENT_COMPARISON_SCHEMA_VERSION] = (
        AGENT_COMPARISON_SCHEMA_VERSION
    )
    status: Literal["complete"] = "complete"
    annotation_semantics: Literal[ANNOTATION_SEMANTICS] = ANNOTATION_SEMANTICS
    config: SurfaceComparisonConfig = Field(default_factory=SurfaceComparisonConfig)
    packet_count: int = Field(ge=1)
    annotation_status_counts: dict[str, int]
    completed_packet_count: int = Field(ge=0)
    replay_reviewer_ids: list[str] = Field(default_factory=list)
    assessable_family_count: int = Field(ge=0)
    comparable_family_count: int = Field(ge=0)
    comparable_family_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    active_line_count: int = Field(ge=0)
    assigned_active_line_count: int = Field(ge=0)
    active_line_assignment_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    family_counts: BinaryDecisionCounts = Field(default_factory=BinaryDecisionCounts)
    pair_counts: BinaryDecisionCounts = Field(default_factory=BinaryDecisionCounts)
    macro_same_surface_pair_retention: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    macro_different_surface_pair_separation: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    records: list[FamilySurfaceComparison] = Field(default_factory=list)
    agent_annotation_hash_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_protocol_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    agent_annotation_audit_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    baseline_audit_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    review_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    counterfactual_human_gates: dict[str, bool]
    counterfactual_human_thresholds_passed: bool
    decision: Literal["diagnostic_only_human_confirmation_still_required"] = (
        "diagnostic_only_human_confirmation_still_required"
    )
    ai_assisted_annotations_used: Literal[True] = True
    human_annotations_used: Literal[False] = False
    posthoc_source_key_opened: Literal[False] = False
    source_labels_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False
    web_integration_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _diagnostic_closure(self) -> "AgentSurfaceComparisonReport":
        if self.replay_reviewer_ids != sorted(set(self.replay_reviewer_ids)):
            raise ValueError("agent diagnostic replay reviewer IDs must be unique")
        if self.completed_packet_count != len(self.replay_reviewer_ids):
            raise ValueError("completed packet and replay reviewer counts differ")
        if self.counterfactual_human_thresholds_passed != all(
            self.counterfactual_human_gates.values()
        ):
            raise ValueError("counterfactual pass flag disagrees with its gates")
        return self


class AgentDiagnosticReplayAuthorization(_FrozenAgentDiagnosticContract):
    schema_version: Literal[AGENT_AUTHORIZATION_SCHEMA_VERSION] = (
        AGENT_AUTHORIZATION_SCHEMA_VERSION
    )
    diagnostic_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_annotation_hash_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    g1_g4_measurement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_ids: list[str] = Field(min_length=1)
    annotation_semantics: Literal[ANNOTATION_SEMANTICS] = ANNOTATION_SEMANTICS
    ai_assisted_annotations_used: Literal[True] = True
    human_annotations_used: Literal[False] = False
    posthoc_source_key_opened: Literal[False] = False
    source_labels_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False
    web_integration_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _reviewer_closure(self) -> "AgentDiagnosticReplayAuthorization":
        if self.reviewer_ids != sorted(set(self.reviewer_ids)):
            raise ValueError("agent diagnostic reviewer IDs must be unique and sorted")
        return self


class AgentSurfaceConditionedDiagnostic(_FrozenAgentDiagnosticContract):
    schema_version: Literal[AGENT_REPLAY_SCHEMA_VERSION] = AGENT_REPLAY_SCHEMA_VERSION
    annotation_semantics: Literal[ANNOTATION_SEMANTICS] = ANNOTATION_SEMANTICS
    diagnostic_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: SurfaceConditionedReplayResult
    ai_assisted_annotations_used: Literal[True] = True
    human_annotations_used: Literal[False] = False
    posthoc_source_key_opened: Literal[False] = False
    source_labels_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False
    web_integration_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _inner_result_remains_diagnostic(self) -> "AgentSurfaceConditionedDiagnostic":
        if self.result.human_annotations_used:
            raise ValueError("inner replay unexpectedly records human annotations")
        if self.result.origin_scoring_authorized:
            raise ValueError("inner replay unexpectedly authorizes origin scoring")
        if self.result.web_integration_authorized:
            raise ValueError("inner replay unexpectedly authorizes web integration")
        return self


def compare_deterministic_surfaces_with_agent_annotations(
    packets: dict[str, GeometryRelationReviewPacket],
    annotations: dict[str, GeometryRelationAnnotation],
    baselines: dict[str, DeterministicSurfaceBaselineResult],
    *,
    config: SurfaceComparisonConfig = DEFAULT_SURFACE_COMPARISON_CONFIG,
) -> AgentSurfaceComparisonReport:
    """Run the frozen relation metrics with explicit AI-preannotation semantics."""

    reviewer_ids = set(packets)
    if reviewer_ids != set(annotations) or reviewer_ids != set(baselines):
        raise ValueError("packet, annotation and baseline reviewer IDs must match")
    if len(reviewer_ids) != config.expected_packet_count:
        raise ValueError("agent diagnostic packet count differs from its protocol")

    status_counts: dict[str, int] = {}
    for annotation in annotations.values():
        status_counts[annotation.status] = status_counts.get(annotation.status, 0) + 1
    if status_counts.get("pending", 0):
        raise ValueError("agent diagnostic annotations must all be terminal")

    records: list[FamilySurfaceComparison] = []
    family_counts = _MutableBinaryCounts()
    pair_counts = _MutableBinaryCounts()
    active_line_count = 0
    assigned_active_line_count = 0
    comparable_family_count = 0
    assessable_family_count = 0
    completed_packet_count = 0
    replay_reviewer_ids: list[str] = []
    same_surface_rates: list[float] = []
    different_surface_rates: list[float] = []
    canonical_annotations: list[dict[str, object]] = []

    for reviewer_id in sorted(reviewer_ids):
        packet = packets[reviewer_id]
        annotation = annotations[reviewer_id]
        baseline = baselines[reviewer_id]
        _validate_identity_and_baseline(packet, annotation, baseline)
        canonical_annotations.append(
            {
                "reviewer_id": reviewer_id,
                "annotation": annotation.model_dump(mode="json"),
            }
        )
        if annotation.status == "unassessable":
            continue
        completed_packet_count += 1
        replay_reviewer_ids.append(reviewer_id)
        validate_annotation_semantic_closure(packet, annotation)
        review_by_id = {
            review.proposed_family_id: review
            for review in annotation.proposed_family_reviews
        }
        partition_by_id = {
            partition.original_family_id: partition
            for partition in baseline.family_partitions
        }
        for proposal in packet.family_proposals:
            review = review_by_id[proposal.family_id]
            partition = partition_by_id[proposal.family_id]
            if review.verdict == "unassessable":
                records.append(
                    _unassessable_family_record(
                        reviewer_id,
                        proposal.family_id,
                        partition,
                    )
                )
                continue
            assessable_family_count += 1
            if partition.partition_status == "insufficient_support":
                family_counts.abstained += 1
            else:
                comparable_family_count += 1
                _update_family_counts(
                    family_counts,
                    review.verdict,
                    partition.partition_status,
                )
            record = _compare_family_pairs(
                reviewer_id,
                proposal.family_id,
                proposal.member_line_ids,
                review,
                annotation,
                partition,
            )
            records.append(record)
            active_line_count += record.active_member_count
            assigned_active_line_count += record.baseline_assigned_member_count
            pair_counts.add(record.pair_counts)
            if record.same_surface_pair_retention is not None:
                same_surface_rates.append(record.same_surface_pair_retention)
            if record.different_surface_pair_separation is not None:
                different_surface_rates.append(
                    record.different_surface_pair_separation
                )

    comparable_fraction = _rate(comparable_family_count, assessable_family_count)
    assignment_fraction = _rate(assigned_active_line_count, active_line_count)
    macro_same = _mean_or_none(same_surface_rates)
    macro_different = _mean_or_none(different_surface_rates)
    frozen_family_counts = family_counts.freeze()
    gates = {
        "terminal_annotation_ratio_at_least_minimum": True,
        "completed_packet_count_at_least_minimum": (
            completed_packet_count >= config.completed_packet_count_minimum
        ),
        "assessable_family_count_at_least_minimum": (
            assessable_family_count >= config.assessable_family_count_minimum
        ),
        "comparable_family_fraction_at_least_minimum": _at_least(
            comparable_fraction,
            config.comparable_family_fraction_minimum,
        ),
        "active_line_assignment_fraction_at_least_minimum": _at_least(
            assignment_fraction,
            config.active_line_assignment_fraction_minimum,
        ),
        "macro_same_surface_pair_retention_at_least_minimum": _at_least(
            macro_same,
            config.macro_same_surface_pair_retention_minimum,
        ),
        "macro_different_surface_pair_separation_at_least_minimum": _at_least(
            macro_different,
            config.macro_different_surface_pair_separation_minimum,
        ),
        "split_family_recall_at_least_minimum": _at_least(
            frozen_family_counts.positive_recall,
            config.split_family_recall_minimum,
        ),
        "non_split_family_specificity_at_least_minimum": _at_least(
            frozen_family_counts.negative_specificity,
            config.non_split_family_specificity_minimum,
        ),
    }
    return AgentSurfaceComparisonReport(
        config=config,
        packet_count=len(reviewer_ids),
        annotation_status_counts=dict(sorted(status_counts.items())),
        completed_packet_count=completed_packet_count,
        replay_reviewer_ids=replay_reviewer_ids,
        assessable_family_count=assessable_family_count,
        comparable_family_count=comparable_family_count,
        comparable_family_fraction=comparable_fraction,
        active_line_count=active_line_count,
        assigned_active_line_count=assigned_active_line_count,
        active_line_assignment_fraction=assignment_fraction,
        family_counts=frozen_family_counts,
        pair_counts=pair_counts.freeze(),
        macro_same_surface_pair_retention=macro_same,
        macro_different_surface_pair_separation=macro_different,
        records=records,
        agent_annotation_hash_closure_sha256=_canonical_hash(
            canonical_annotations
        ),
        counterfactual_human_gates=gates,
        counterfactual_human_thresholds_passed=all(gates.values()),
    )


def build_agent_diagnostic_replay_authorization(
    comparison_report: dict[str, object],
    *,
    comparison_report_sha256: str,
    g1_g4_measurement_sha256: str,
) -> AgentDiagnosticReplayAuthorization:
    """Authorize diagnostics without converting AI annotations into human gates."""

    if comparison_report.get("schema_version") != AGENT_COMPARISON_SCHEMA_VERSION:
        raise ValueError("unexpected agent surface comparison schema")
    if comparison_report.get("status") != "complete":
        raise ValueError("agent surface comparison is not complete")
    if comparison_report.get("decision") != (
        "diagnostic_only_human_confirmation_still_required"
    ):
        raise ValueError("agent surface comparison is not diagnostic-only")
    if comparison_report.get("annotation_semantics") != ANNOTATION_SEMANTICS:
        raise ValueError("agent surface comparison annotation semantics changed")
    required_flags = {
        "ai_assisted_annotations_used": True,
        "human_annotations_used": False,
        "posthoc_source_key_opened": False,
        "source_labels_used": False,
        "origin_scoring_authorized": False,
        "web_integration_authorized": False,
    }
    for name, expected in required_flags.items():
        if comparison_report.get(name) is not expected:
            raise ValueError(f"agent comparison violates diagnostic field {name}")
    reviewer_ids = comparison_report.get("replay_reviewer_ids")
    if not isinstance(reviewer_ids, list) or not reviewer_ids:
        raise ValueError("agent comparison has no diagnostic replay reviewers")
    return AgentDiagnosticReplayAuthorization(
        diagnostic_protocol_sha256=str(
            comparison_report["diagnostic_protocol_sha256"]
        ),
        comparison_report_sha256=comparison_report_sha256,
        baseline_audit_sha256=str(comparison_report["baseline_audit_sha256"]),
        agent_annotation_hash_closure_sha256=str(
            comparison_report["agent_annotation_hash_closure_sha256"]
        ),
        g1_g4_measurement_sha256=g1_g4_measurement_sha256,
        reviewer_ids=sorted(str(value) for value in reviewer_ids),
    )


def assess_agent_surface_conditioned_g1_g4(
    canonical_rgb: np.ndarray,
    measurement: GeometryMeasurementV2Result,
    packet: GeometryRelationReviewPacket,
    baseline: DeterministicSurfaceBaselineResult,
    authorization: AgentDiagnosticReplayAuthorization,
) -> AgentSurfaceConditionedDiagnostic:
    """Wrap the frozen surface-conditioned replay with explicit AI semantics."""

    result = assess_surface_conditioned_g1_g4(
        canonical_rgb,
        measurement,
        packet,
        baseline,
        authorization,  # type: ignore[arg-type]
    )
    return AgentSurfaceConditionedDiagnostic(
        diagnostic_protocol_sha256=authorization.diagnostic_protocol_sha256,
        result=result,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None
