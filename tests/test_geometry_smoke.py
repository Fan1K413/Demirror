from pathlib import Path
import json

from PIL import Image, PngImagePlugin

from image_trust.geometry.line_backend import BackendUnavailableError, resolve_backend
from image_trust.pipeline import analyze_image
from image_trust.schemas import Direction, LineBackendConfig, RunStatus
from image_trust.utils.config import load_config
from tests.helpers import write_solid_image, write_structured_image


def test_opencv_pipeline_writes_required_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "structure.png"
    output_dir = tmp_path / "output"
    write_structured_image(input_path)
    config = load_config(Path("configs/p0.yaml"))
    result = analyze_image(input_path, config, output_dir)
    assert result.evidence.run_status == RunStatus.OK
    assert result.evidence.direction == Direction.NEUTRAL
    assert result.evidence.features["line_count"] > 0
    assert result.evidence.raw_score is None
    assert result.evidence.features["coverage_definition"]["version"] == "p0-coverage-v1"
    assert result.run.config_digest
    for name in (
        "result.json",
        "lines.json",
        "lines_overlay.png",
        "anomalous_lines_overlay.png",
    ):
        assert (output_dir / name).is_file()
    with Image.open(output_dir / "lines_overlay.png") as overlay:
        assert overlay.size == result.input.canonical_size
    with Image.open(output_dir / "anomalous_lines_overlay.png") as overlay:
        assert overlay.size == result.input.canonical_size
    lines = json.loads((output_dir / "lines.json").read_text(encoding="utf-8"))
    assert [line["line_id"] for line in lines] == [
        f"l{index:06d}" for index in range(1, len(lines) + 1)
    ]


def test_low_geometry_image_does_not_emit_ai_direction(tmp_path: Path) -> None:
    input_path = tmp_path / "solid.png"
    write_solid_image(input_path)
    config = load_config(Path("configs/p0.yaml"))
    result = analyze_image(input_path, config, tmp_path / "output")
    assert result.evidence.run_status == RunStatus.NOT_APPLICABLE
    assert result.evidence.direction == Direction.NEUTRAL
    assert result.evidence.observation.value == "not_observed"
    assert result.evidence.applicability < 0.45
    assert result.evidence.reliability == 0.0
    assert result.evidence.features["line_count"] == 0


def test_repeated_runs_keep_structured_measurements_stable(tmp_path: Path) -> None:
    input_path = tmp_path / "structure.png"
    write_structured_image(input_path)
    config = load_config(Path("configs/p0.yaml"))
    first = analyze_image(input_path, config, tmp_path / "first")
    second = analyze_image(input_path, config, tmp_path / "second")
    assert first.run.deterministic_seed == second.run.deterministic_seed
    assert first.evidence.features == second.evidence.features

    def normalized_result(path: Path) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run"].pop("created_at_utc")
        payload["diagnostics"].pop("timing_ms")
        return payload

    assert normalized_result(tmp_path / "first" / "result.json") == normalized_result(
        tmp_path / "second" / "result.json"
    )
    assert (tmp_path / "first" / "lines.json").read_bytes() == (
        tmp_path / "second" / "lines.json"
    ).read_bytes()
    assert (tmp_path / "first" / "lines_overlay.png").read_bytes() == (
        tmp_path / "second" / "lines_overlay.png"
    ).read_bytes()
    assert (tmp_path / "first" / "anomalous_lines_overlay.png").read_bytes() == (
        tmp_path / "second" / "anomalous_lines_overlay.png"
    ).read_bytes()


def test_explicit_deeplsd_without_weights_never_falls_back() -> None:
    config = LineBackendConfig(name="deeplsd", deeplsd_weights=None)
    try:
        resolve_backend(config)
    except BackendUnavailableError as raised:
        assert raised.backend == "deeplsd"
    else:
        raise AssertionError("DeepLSD must be explicitly unavailable without weights.")


def test_auto_fallback_is_recorded_as_structured_run_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "structure.png"
    write_structured_image(input_path)
    config = load_config(Path("configs/p0.yaml"))
    config = config.model_copy(
        update={
            "line_backend": config.line_backend.model_copy(
                update={"name": "auto", "allow_fallback": True}
            )
        }
    )
    result = analyze_image(input_path, config, tmp_path / "output")
    assert result.run.requested_backend == "auto"
    assert result.run.resolved_backend == "opencv_lsd"
    assert result.run.fallback_reason == "deeplsd_unavailable"
    assert result.run.runtime_environment["python_version"]


def test_special_imaging_metadata_is_not_applicable(tmp_path: Path) -> None:
    input_path = tmp_path / "panorama.png"
    image = Image.new("RGB", (640, 480), (245, 245, 245))
    info = PngImagePlugin.PngInfo()
    info.add_text("XMP", "GPano:ProjectionType=equirectangular")
    image.save(input_path, pnginfo=info)
    config = load_config(Path("configs/p0.yaml"))
    result = analyze_image(input_path, config, tmp_path / "output")
    assert result.evidence.run_status == RunStatus.NOT_APPLICABLE
    assert result.evidence.observation.value == "not_observed"
    assert "panorama_metadata_detected" in result.evidence.limitations
