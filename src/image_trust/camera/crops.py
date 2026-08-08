"""Deterministic P1 crop planning and coordinate conversion."""

from __future__ import annotations

from collections.abc import Iterable

from image_trust.camera.contracts import (
    CameraCropProtocolConfig,
    CameraEstimate,
    CoordinateSpace,
    CropSpec,
    FieldOfViewOrFocal,
    HorizonLine,
    IntrinsicKind,
    Matrix3x3,
)
from image_trust.schemas import Point


def plan_overlapping_crops(
    canonical_size: tuple[int, int],
    config: CameraCropProtocolConfig,
) -> list[CropSpec]:
    """Plan 4, 6, or 8 deterministic overlapping square crop candidates.

    The first four positions straddle the image centre, covering the four
    centre-relative quadrants.  Their neighbour separation encodes the
    configured overlap target.  Extra positions improve centre and axial
    coverage.  Extreme configuration/aspect-ratio combinations can be
    clamped to the image boundary; that fact is preserved per crop rather than
    being silently treated as the requested overlap.
    """

    width, height = canonical_size
    if width <= 0 or height <= 0:
        raise ValueError("canonical_size must contain positive dimensions.")
    side = max(1, round(min(width, height) * config.side_fraction_of_short_edge))
    side = min(side, width, height)
    separation = side * (1.0 - config.target_overlap_fraction)
    offset = separation / 2.0
    centres = _candidate_centres(config.crop_count, width, height, offset)
    crops: list[CropSpec] = []
    for index, requested_center in enumerate(centres, start=1):
        x_float = requested_center.x - side / 2.0
        y_float = requested_center.y - side / 2.0
        x = min(max(round(x_float), 0), width - side)
        y = min(max(round(y_float), 0), height - side)
        clamped = abs(x - x_float) > 0.51 or abs(y - y_float) > 0.51
        crops.append(
            CropSpec(
                crop_id=f"crop-{index:02d}",
                x=x,
                y=y,
                width=side,
                height=side,
                crop_to_canonical=Matrix3x3.translation(float(x), float(y)),
                requested_center=requested_center,
                clamped_to_image_bounds=clamped,
            )
        )
    return crops


def map_estimate_to_canonical(
    estimate: CameraEstimate,
    crop: CropSpec | None,
) -> CameraEstimate:
    """Map crop-local principal point/horizon (and focal pixels) to canonical.

    Camera-level VFOV, roll, and pitch are retained.  ``focal_px`` is mapped
    only through an isotropic affine scale; a non-affine future protocol must
    leave the intrinsic incomparable rather than inventing a conversion.
    """

    if estimate.coordinate_space is CoordinateSpace.CANONICAL:
        return estimate
    transform = crop.crop_to_canonical if crop is not None else Matrix3x3.translation(0.0, 0.0)
    limitations = list(estimate.limitations)
    principal_point = (
        transform.map_point(estimate.principal_point)
        if estimate.principal_point is not None
        else None
    )
    horizon = (
        HorizonLine(
            p1=transform.map_point(estimate.horizon.p1),
            p2=transform.map_point(estimate.horizon.p2),
        )
        if estimate.horizon is not None
        else None
    )
    intrinsic = estimate.vfov_or_focal
    if intrinsic is not None and intrinsic.kind is IntrinsicKind.FOCAL_PX:
        try:
            intrinsic = FieldOfViewOrFocal(
                kind=intrinsic.kind,
                value=intrinsic.value * transform.isotropic_scale,
                reference=intrinsic.reference,
            )
        except ValueError:
            intrinsic = None
            limitations.append("focal_px_not_mapped_by_nonisotropic_transform")
    return estimate.model_copy(
        update={
            "principal_point": principal_point,
            "horizon": horizon,
            "vfov_or_focal": intrinsic,
            "coordinate_space": CoordinateSpace.CANONICAL,
            "limitations": sorted(set(limitations)),
        }
    )


def crop_plan_limitations(crops: Iterable[CropSpec]) -> list[str]:
    """Return any protocol deviations that a caller must disclose."""

    if any(crop.clamped_to_image_bounds for crop in crops):
        return ["crop_window_clamped_to_image_bounds"]
    return []


def _candidate_centres(
    crop_count: int,
    width: int,
    height: int,
    offset: float,
) -> list[Point]:
    centre = Point(x=width / 2.0, y=height / 2.0)
    nw = Point(x=centre.x - offset, y=centre.y - offset)
    ne = Point(x=centre.x + offset, y=centre.y - offset)
    sw = Point(x=centre.x - offset, y=centre.y + offset)
    se = Point(x=centre.x + offset, y=centre.y + offset)
    if crop_count == 4:
        return [nw, ne, sw, se]
    if crop_count == 6:
        # The fifth candidate samples the centre.  The sixth extends along the
        # long axis, avoiding a hidden portrait/landscape preference.
        if width >= height:
            extension = Point(x=centre.x - offset, y=centre.y)
        else:
            extension = Point(x=centre.x, y=centre.y - offset)
        return [nw, ne, sw, se, centre, extension]
    if crop_count == 8:
        north = Point(x=centre.x, y=centre.y - offset)
        south = Point(x=centre.x, y=centre.y + offset)
        west = Point(x=centre.x - offset, y=centre.y)
        east = Point(x=centre.x + offset, y=centre.y)
        return [nw, north, ne, west, east, sw, south, se]
    raise ValueError("crop_count must be one of 4, 6, or 8.")
