"""Source-neutral comparison of deterministic and independently reviewed surfaces."""

from __future__ import annotations

import hashlib
import json
import statistics
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.geometry_ai.deterministic_surfaces import (
    DeterministicFamilyPartition,
    DeterministicSurfaceBaselineResult,
)
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
    ProposedFamilyReview,
)
from image_trust.geometry_ai.relation_validation import (
    validate_annotation_semantic_closure,
)


QUALITY_RECEIPT_SCHEMA_VERSION = "geometry-human-relation-quality-receipt-v1"
COMPARISON_SCHEMA_VERSION = "geometry-surface-human-comparison-v1"
REQUIRED_QUALITY_GATES = frozenset(
    {
        "completed_unique_ratio_at_least_0_75",
        "completed_duplicate_pairs_at_least_3",
        "family_decision_agreement_at_least_0_80",
        "surface_line_pair_agreement_at_least_0_80",
    }
)


class _FrozenComparisonContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HumanRelationQualityReceipt(_FrozenComparisonContract):
    schema_version: Literal[QUALITY_RECEIPT_SCHEMA_VERSION] = (
        QUALITY_RECEIPT_SCHEMA_VERSION
    )
    input_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_count: int = Field(ge=1)
    unique_source_count: int = Field(ge=1)
    completed_unique_count: int = Field(ge=0)
    hidden_duplicate_group_count: int = Field(ge=1)
    completed_duplicate_pair_count: int = Field(ge=0)
    family_decision_agreement: float = Field(ge=0.0, le=1.0)
    surface_line_pair_agreement: float = Field(ge=0.0, le=1.0)
    gates: dict[str, bool]
    passed: bool
    source_details_transferred: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _quality_gate_closure(self) -> "HumanRelationQualityReceipt":
        if set(self.gates) != REQUIRED_QUALITY_GATES:
            raise ValueError("quality receipt gates do not match the frozen contract")
        if self.completed_unique_count > self.unique_source_count:
            raise ValueError("completed unique reviews exceed the unique source count")
        if self.completed_duplicate_pair_count > self.hidden_duplicate_group_count:
            raise ValueError("completed duplicate pairs exceed hidden duplicate groups")
        if self.packet_count != self.unique_source_count + self.hidden_duplicate_group_count:
            raise ValueError("quality receipt packet and source counts do not close")
        if self.passed != all(self.gates.values()):
            raise ValueError("quality receipt passed flag disagrees with its gates")
        expected_gates = {
            "completed_unique_ratio_at_least_0_75": (
                self.completed_unique_count / self.unique_source_count >= 0.75
            ),
            "completed_duplicate_pairs_at_least_3": (
                self.completed_duplicate_pair_count >= 3
            ),
            "family_decision_agreement_at_least_0_80": (
                self.family_decision_agreement >= 0.80
            ),
            "surface_line_pair_agreement_at_least_0_80": (
                self.surface_line_pair_agreement >= 0.80
            ),
        }
        if self.gates != expected_gates:
            raise ValueError("quality receipt gates disagree with their measurements")
        return self


class SurfaceComparisonConfig(_FrozenComparisonContract):
    expected_packet_count: int = Field(default=36, ge=1)
    terminal_annotation_ratio_minimum: float = Field(default=1.0, ge=0.0, le=1.0)
    completed_packet_count_minimum: int = Field(default=24, ge=1)
    assessable_family_count_minimum: int = Field(default=48, ge=1)
    comparable_family_fraction_minimum: float = Field(default=0.75, ge=0.0, le=1.0)
    active_line_assignment_fraction_minimum: float = Field(default=0.70, ge=0.0, le=1.0)
    macro_same_surface_pair_retention_minimum: float = Field(
        default=0.70, ge=0.0, le=1.0
    )
    macro_different_surface_pair_separation_minimum: float = Field(
        default=0.70, ge=0.0, le=1.0
    )
    split_family_recall_minimum: float = Field(default=0.60, ge=0.0, le=1.0)
    non_split_family_specificity_minimum: float = Field(
        default=0.70, ge=0.0, le=1.0
    )


DEFAULT_SURFACE_COMPARISON_CONFIG = SurfaceComparisonConfig()


class BinaryDecisionCounts(_FrozenComparisonContract):
    true_positive: int = Field(default=0, ge=0)
    true_negative: int = Field(default=0, ge=0)
    false_positive: int = Field(default=0, ge=0)
    false_negative: int = Field(default=0, ge=0)
    abstained: int = Field(default=0, ge=0)

    @property
    def positive_recall(self) -> float | None:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else None

    @property
    def negative_specificity(self) -> float | None:
        denominator = self.true_negative + self.false_positive
        return self.true_negative / denominator if denominator else None


class FamilySurfaceComparison(_FrozenComparisonContract):
    reviewer_id: str
    family_id: str
    human_verdict: Literal[
        "coherent_within_surface",
        "split_across_surfaces",
        "geometry_inconsistent_within_surface",
        "unassessable",
    ]
    baseline_partition_status: Literal[
        "split_candidate",
        "single_surface_candidate",
        "insufficient_support",
    ]
    active_member_count: int = Field(ge=0)
    human_outlier_count: int = Field(ge=0)
    baseline_assigned_member_count: int = Field(ge=0)
    pair_counts: BinaryDecisionCounts = Field(default_factory=BinaryDecisionCounts)
    same_surface_pair_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    different_surface_pair_separation: float | None = Field(
        default=None, ge=0.0, le=1.0
    )


class GeometrySurfaceComparisonReport(_FrozenComparisonContract):
    schema_version: Literal[COMPARISON_SCHEMA_VERSION] = COMPARISON_SCHEMA_VERSION
    status: Literal["waiting_for_human_annotations", "complete"]
    config: SurfaceComparisonConfig = Field(default_factory=SurfaceComparisonConfig)
    packet_count: int = Field(ge=0)
    annotation_status_counts: dict[str, int]
    terminal_annotation_ratio: float = Field(ge=0.0, le=1.0)
    completed_packet_count: int = Field(ge=0)
    replay_reviewer_ids: list[str] = Field(default_factory=list)
    assessable_family_count: int = Field(ge=0)
    comparable_family_count: int = Field(ge=0)
    comparable_family_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    active_line_count: int = Field(ge=0)
    assigned_active_line_count: int = Field(ge=0)
    active_line_assignment_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    family_counts: BinaryDecisionCounts = Field(default_factory=BinaryDecisionCounts)
    pair_counts: BinaryDecisionCounts = Field(default_factory=BinaryDecisionCounts)
    macro_same_surface_pair_retention: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    macro_different_surface_pair_separation: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    records: list[FamilySurfaceComparison] = Field(default_factory=list)
    human_annotation_hash_closure_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    quality_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    continuation_protocol_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    baseline_audit_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    review_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    gates: dict[str, bool]
    passed: bool
    decision: Literal[
        "wait_for_independent_human_annotations",
        "eligible_for_surface_conditioned_g1_g4_replay_only",
        "keep_as_review_visualization_without_tuning",
    ]
    source_key_opened: Literal[False] = False
    source_labels_used: Literal[False] = False
    ai_assisted_annotations_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False
    web_integration_authorized: Literal[False] = False


def extract_human_quality_receipt(
    pilot_audit: dict[str, object],
    *,
    input_audit_sha256: str,
) -> HumanRelationQualityReceipt:
    """Reduce the posthoc audit to source-neutral review-quality fields."""

    if pilot_audit.get("schema_version") != "geometry-semantic-relation-pilot-audit-v1":
        raise ValueError("unexpected human relation pilot audit schema")
    if pilot_audit.get("status") != "complete":
        raise ValueError("human relation pilot audit is not complete")
    if pilot_audit.get("source_key_opened") is not True:
        raise ValueError("human relation quality was not evaluated with hidden duplicates")
    if pilot_audit.get("origin_scoring_authorized") is not False:
        raise ValueError("human relation pilot audit unexpectedly authorizes scoring")
    quality = pilot_audit.get("quality")
    if not isinstance(quality, dict):
        raise ValueError("human relation pilot audit has no quality block")
    gates = quality.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("human relation pilot audit has no quality gates")
    numeric_quality_fields = (
        "family_decision_agreement",
        "surface_line_pair_agreement",
    )
    if any(
        not isinstance(quality.get(field), (int, float))
        or isinstance(quality.get(field), bool)
        for field in numeric_quality_fields
    ):
        raise ValueError("human relation pilot audit has invalid quality measurements")
    return HumanRelationQualityReceipt(
        input_audit_sha256=input_audit_sha256,
        packet_count=int(pilot_audit.get("packet_count", 0)),
        unique_source_count=int(pilot_audit.get("unique_source_count", 0)),
        completed_unique_count=int(quality.get("completed_unique_count", 0)),
        hidden_duplicate_group_count=int(
            quality.get("hidden_duplicate_group_count", 0)
        ),
        completed_duplicate_pair_count=int(
            quality.get("completed_duplicate_pair_count", 0)
        ),
        family_decision_agreement=float(quality["family_decision_agreement"]),
        surface_line_pair_agreement=float(quality["surface_line_pair_agreement"]),
        gates={str(key): bool(value) for key, value in gates.items()},
        passed=bool(quality.get("passed")),
    )


def compare_deterministic_surfaces_with_human(
    packets: dict[str, GeometryRelationReviewPacket],
    annotations: dict[str, GeometryRelationAnnotation],
    baselines: dict[str, DeterministicSurfaceBaselineResult],
    *,
    quality_receipt: HumanRelationQualityReceipt | None,
    config: SurfaceComparisonConfig = DEFAULT_SURFACE_COMPARISON_CONFIG,
) -> GeometrySurfaceComparisonReport:
    """Compare source-blind family and line-pair relations without source data."""

    reviewer_ids = set(packets)
    if reviewer_ids != set(annotations) or reviewer_ids != set(baselines):
        raise ValueError("packet, annotation and baseline reviewer IDs must match exactly")
    if len(reviewer_ids) != config.expected_packet_count:
        raise ValueError("comparison packet count differs from the frozen configuration")

    status_counts: dict[str, int] = {}
    for annotation in annotations.values():
        status_counts[annotation.status] = status_counts.get(annotation.status, 0) + 1
    pending_count = status_counts.get("pending", 0)
    terminal_ratio = (len(reviewer_ids) - pending_count) / max(len(reviewer_ids), 1)
    if pending_count:
        return _waiting_report(config, len(reviewer_ids), status_counts, terminal_ratio)
    if quality_receipt is None or not quality_receipt.passed:
        raise ValueError("a passing source-neutral human quality receipt is required")
    if quality_receipt.packet_count != len(reviewer_ids):
        raise ValueError("quality receipt packet count differs from the comparison cohort")

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
                    _unassessable_family_record(reviewer_id, proposal.family_id, partition)
                )
                continue
            assessable_family_count += 1
            if partition.partition_status == "insufficient_support":
                family_counts.abstained += 1
            else:
                comparable_family_count += 1
                _update_family_counts(family_counts, review.verdict, partition.partition_status)
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
        "human_quality_receipt_passed": quality_receipt.passed,
        "terminal_annotation_ratio_at_least_minimum": (
            terminal_ratio >= config.terminal_annotation_ratio_minimum
        ),
        "completed_packet_count_at_least_minimum": (
            completed_packet_count >= config.completed_packet_count_minimum
        ),
        "assessable_family_count_at_least_minimum": (
            assessable_family_count >= config.assessable_family_count_minimum
        ),
        "comparable_family_fraction_at_least_minimum": _at_least(
            comparable_fraction, config.comparable_family_fraction_minimum
        ),
        "active_line_assignment_fraction_at_least_minimum": _at_least(
            assignment_fraction, config.active_line_assignment_fraction_minimum
        ),
        "macro_same_surface_pair_retention_at_least_minimum": _at_least(
            macro_same, config.macro_same_surface_pair_retention_minimum
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
    passed = all(gates.values())
    return GeometrySurfaceComparisonReport(
        status="complete",
        config=config,
        packet_count=len(reviewer_ids),
        annotation_status_counts=dict(sorted(status_counts.items())),
        terminal_annotation_ratio=terminal_ratio,
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
        human_annotation_hash_closure_sha256=_canonical_hash(canonical_annotations),
        quality_receipt_sha256=_canonical_hash(
            quality_receipt.model_dump(mode="json")
        ),
        gates=gates,
        passed=passed,
        decision=(
            "eligible_for_surface_conditioned_g1_g4_replay_only"
            if passed
            else "keep_as_review_visualization_without_tuning"
        ),
    )


def _waiting_report(
    config: SurfaceComparisonConfig,
    packet_count: int,
    status_counts: dict[str, int],
    terminal_ratio: float,
) -> GeometrySurfaceComparisonReport:
    return GeometrySurfaceComparisonReport(
        status="waiting_for_human_annotations",
        config=config,
        packet_count=packet_count,
        annotation_status_counts=dict(sorted(status_counts.items())),
        terminal_annotation_ratio=terminal_ratio,
        completed_packet_count=status_counts.get("completed", 0),
        assessable_family_count=0,
        comparable_family_count=0,
        active_line_count=0,
        assigned_active_line_count=0,
        gates={
            "human_quality_receipt_passed": False,
            "terminal_annotation_ratio_at_least_minimum": False,
            "completed_packet_count_at_least_minimum": False,
            "assessable_family_count_at_least_minimum": False,
            "comparable_family_fraction_at_least_minimum": False,
            "active_line_assignment_fraction_at_least_minimum": False,
            "macro_same_surface_pair_retention_at_least_minimum": False,
            "macro_different_surface_pair_separation_at_least_minimum": False,
            "split_family_recall_at_least_minimum": False,
            "non_split_family_specificity_at_least_minimum": False,
        },
        passed=False,
        decision="wait_for_independent_human_annotations",
    )


def _validate_identity_and_baseline(
    packet: GeometryRelationReviewPacket,
    annotation: GeometryRelationAnnotation,
    baseline: DeterministicSurfaceBaselineResult,
) -> None:
    if packet.reviewer_id != annotation.reviewer_id:
        raise ValueError("annotation reviewer ID differs from its packet")
    if packet.reviewer_id != baseline.reviewer_id:
        raise ValueError("baseline reviewer ID differs from its packet")
    if baseline.canonical_size != packet.canonical_size:
        raise ValueError("baseline canonical size differs from its packet")
    if baseline.status != "available":
        raise ValueError("comparison requires an available deterministic baseline")
    if baseline.source_labels_used or baseline.origin_scoring_authorized:
        raise ValueError("comparison baseline unexpectedly contains source authorization")
    packet_families = [family.family_id for family in packet.family_proposals]
    baseline_families = [
        partition.original_family_id for partition in baseline.family_partitions
    ]
    if packet_families != baseline_families:
        raise ValueError("baseline family order differs from its packet")


def _unassessable_family_record(
    reviewer_id: str,
    family_id: str,
    partition: DeterministicFamilyPartition,
) -> FamilySurfaceComparison:
    return FamilySurfaceComparison(
        reviewer_id=reviewer_id,
        family_id=family_id,
        human_verdict="unassessable",
        baseline_partition_status=partition.partition_status,
        active_member_count=0,
        human_outlier_count=0,
        baseline_assigned_member_count=0,
    )


def _update_family_counts(
    counts: "_MutableBinaryCounts",
    human_verdict: str,
    baseline_status: str,
) -> None:
    human_positive = human_verdict == "split_across_surfaces"
    baseline_positive = baseline_status == "split_candidate"
    counts.observe(human_positive, baseline_positive)


def _compare_family_pairs(
    reviewer_id: str,
    family_id: str,
    original_members: list[str],
    review: ProposedFamilyReview,
    annotation: GeometryRelationAnnotation,
    partition: DeterministicFamilyPartition,
) -> FamilySurfaceComparison:
    active_members = [
        line_id
        for line_id in original_members
        if line_id not in set(review.outlier_line_ids)
    ]
    human_membership = {line_id: set() for line_id in active_members}
    referenced_surfaces = set(review.surface_ids)
    for surface in annotation.surfaces:
        if surface.surface_id not in referenced_surfaces:
            continue
        for line_id in set(surface.line_ids) & set(active_members):
            human_membership[line_id].add(surface.surface_id)
    baseline_membership = {line_id: set() for line_id in active_members}
    for subfamily in partition.surface_subfamilies:
        for line_id in set(subfamily.member_line_ids) & set(active_members):
            baseline_membership[line_id].add(subfamily.surface_id)
    assigned_members = {
        line_id for line_id, surfaces in baseline_membership.items() if surfaces
    }
    pair_counts = _MutableBinaryCounts()
    for first, second in combinations(active_members, 2):
        if first not in assigned_members or second not in assigned_members:
            pair_counts.abstained += 1
            continue
        human_same = bool(human_membership[first] & human_membership[second])
        baseline_same = bool(baseline_membership[first] & baseline_membership[second])
        pair_counts.observe(human_same, baseline_same)
    frozen = pair_counts.freeze()
    return FamilySurfaceComparison(
        reviewer_id=reviewer_id,
        family_id=family_id,
        human_verdict=review.verdict,
        baseline_partition_status=partition.partition_status,
        active_member_count=len(active_members),
        human_outlier_count=len(review.outlier_line_ids),
        baseline_assigned_member_count=len(assigned_members),
        pair_counts=frozen,
        same_surface_pair_retention=frozen.positive_recall,
        different_surface_pair_separation=frozen.negative_specificity,
    )


class _MutableBinaryCounts:
    def __init__(self) -> None:
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.abstained = 0

    def observe(self, reference_positive: bool, predicted_positive: bool) -> None:
        if reference_positive and predicted_positive:
            self.true_positive += 1
        elif reference_positive:
            self.false_negative += 1
        elif predicted_positive:
            self.false_positive += 1
        else:
            self.true_negative += 1

    def add(self, other: BinaryDecisionCounts) -> None:
        self.true_positive += other.true_positive
        self.true_negative += other.true_negative
        self.false_positive += other.false_positive
        self.false_negative += other.false_negative
        self.abstained += other.abstained

    def freeze(self) -> BinaryDecisionCounts:
        return BinaryDecisionCounts(
            true_positive=self.true_positive,
            true_negative=self.true_negative,
            false_positive=self.false_positive,
            false_negative=self.false_negative,
            abstained=self.abstained,
        )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
