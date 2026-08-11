from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from image_trust.watermark import trustmark


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "trustmark"
MODEL_PATH = REPOSITORY_ROOT / "weights" / "trustmark" / "Q" / "decoder_Q.onnx"


def _bypass_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trustmark, "_dependency_limitation", lambda: None)
    monkeypatch.setattr(trustmark, "_sha256", lambda _path: trustmark.TRUSTMARK_Q_MODEL_SHA256)


def test_positive_identifier_stays_neutral_and_withholds_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "decoder_Q.onnx"
    model.write_bytes(b"test model")
    _bypass_readiness(monkeypatch)
    payload = "10" * 30 + "1"
    monkeypatch.setattr(
        trustmark,
        "_decode_isolated",
        lambda *_args, **_kwargs: {
            "detected": True,
            "payload_bits": payload,
            "schema": "BCH_5",
        },
    )

    result = trustmark.detect_trustmark_q(FIXTURE_ROOT / "ufo_240_Q.png", model_path=model)

    assert result.run_status == "ok"
    assert result.observation == "positive"
    assert result.evidence_class == "unverified_identifier"
    assert result.direction == "neutral"
    assert result.strength == "none"
    assert result.decision_eligible is False
    assert result.payload.present is True
    assert result.payload.payload_schema == "BCH_5"
    assert result.payload.bit_length == 61
    assert result.payload.sha256 == hashlib.sha256(payload.encode("ascii")).hexdigest()
    dumped = result.model_dump_json()
    assert payload not in dumped


def test_negative_observation_is_neutral(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "decoder_Q.onnx"
    model.write_bytes(b"test model")
    _bypass_readiness(monkeypatch)
    monkeypatch.setattr(
        trustmark,
        "_decode_isolated",
        lambda *_args, **_kwargs: {
            "detected": False,
            "payload_bits": "",
            "schema": None,
        },
    )

    result = trustmark.detect_trustmark_q(FIXTURE_ROOT / "ufo_240.jpg", model_path=model)

    assert result.run_status == "ok"
    assert result.observation == "negative"
    assert result.evidence_class == "none"
    assert result.direction == "neutral"
    assert result.decision_eligible is False
    assert result.payload.present is False


def test_missing_or_changed_model_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trustmark, "_dependency_limitation", lambda: None)
    missing = trustmark.detect_trustmark_q(
        FIXTURE_ROOT / "ufo_240.jpg",
        model_path=tmp_path / "missing.onnx",
    )
    assert missing.run_status == "unavailable"
    assert "trustmark_q_model_not_available" in missing.limitations

    changed_model = tmp_path / "changed.onnx"
    changed_model.write_bytes(b"changed")
    changed = trustmark.detect_trustmark_q(
        FIXTURE_ROOT / "ufo_240.jpg",
        model_path=changed_model,
    )
    assert changed.run_status == "unavailable"
    assert "trustmark_q_model_sha256_mismatch" in changed.limitations


def test_short_image_is_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "decoder_Q.onnx"
    model.write_bytes(b"test model")
    image_path = tmp_path / "short.png"
    Image.new("RGB", (320, 149), "white").save(image_path)
    _bypass_readiness(monkeypatch)

    result = trustmark.detect_trustmark_q(image_path, model_path=model)

    assert result.run_status == "not_applicable"
    assert result.observation == "not_observed"
    assert "trustmark_q_short_side_below_150" in result.limitations


def test_worker_failure_is_contained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "decoder_Q.onnx"
    model.write_bytes(b"test model")
    _bypass_readiness(monkeypatch)

    def fail(*_args, **_kwargs):
        raise trustmark.TrustMarkUnavailableError("trustmark_q_worker_timed_out")

    monkeypatch.setattr(trustmark, "_decode_isolated", fail)
    result = trustmark.detect_trustmark_q(FIXTURE_ROOT / "ufo_240.jpg", model_path=model)

    assert result.run_status == "failed"
    assert result.observation == "not_observed"
    assert result.errors[0]["code"] == "trustmark_q_worker_timed_out"


def test_official_fixtures_match_recorded_hashes() -> None:
    manifest = json.loads((FIXTURE_ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        contents = (FIXTURE_ROOT / item["path"]).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == item["sha256"]


def test_bch5_acceptance_rejects_more_than_three_corrected_bits() -> None:
    bchlib = pytest.importorskip("bchlib")
    payload = "10" * 30 + "1"
    padded = payload + "0" * (-len(payload) % 8)
    data = bytearray(int(padded[index : index + 8], 2) for index in range(0, len(padded), 8))
    encoder = bchlib.BCH(5, 137)
    ecc = "".join(f"{byte:08b}" for byte in encoder.encode(data))[:35]
    packet = payload + ecc + "0001"

    def flipped(count: int) -> str:
        values = list(packet)
        for index in range(count):
            values[index] = "0" if values[index] == "1" else "1"
        return "".join(values)

    accepted = trustmark._decode_ecc(flipped(3))
    rejected = trustmark._decode_ecc(flipped(4))

    assert accepted["detected"] is True
    assert accepted["corrected_bits"] == 3
    assert accepted["payload_bits"] == payload
    assert rejected["detected"] is False


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="optional pinned TrustMark Q model not installed")
def test_official_model_decodes_only_the_watermarked_fixture() -> None:
    positive = trustmark.detect_trustmark_q(FIXTURE_ROOT / "ufo_240_Q.png")
    negative = trustmark.detect_trustmark_q(FIXTURE_ROOT / "ufo_240.jpg")

    assert positive.run_status == "ok"
    assert positive.observation == "positive"
    assert positive.evidence_class == "unverified_identifier"
    assert positive.payload.payload_schema == "BCH_5"
    assert positive.payload.bit_length == 61
    assert negative.run_status == "ok"
    assert negative.observation == "negative"


def test_committed_audit_supports_the_operating_policy() -> None:
    audit = json.loads(
        (REPOSITORY_ROOT / "models" / "implicit_watermark_trustmark_q_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert audit["model"]["sha256"] == trustmark.TRUSTMARK_Q_MODEL_SHA256
    assert audit["acceptance_rule"]["accepted_schema"] == "BCH_5"
    assert audit["acceptance_rule"]["maximum_accepted_corrected_bits"] == 3
    assert audit["acceptance_rule"]["permissive_schema_fallback"] is False
    assert audit["official_fixture_screen"]["positive_detected"] is True
    assert audit["official_fixture_screen"]["negative_detected"] is False
    assert all(
        item["detected"]
        for item in audit["official_fixture_screen"]["transforms"].values()
    )
    assert audit["negative_screen"]["eligible_images"] >= 3_000
    assert audit["negative_screen"]["positive_matches"] == 0
    assert audit["decision_policy"] == {
        "decision_eligible": False,
        "direction": "neutral",
        "evidence_class": "unverified_identifier",
        "raw_payload_returned": False,
    }
