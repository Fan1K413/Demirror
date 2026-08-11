from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request

from PIL import Image

from image_trust.watermark.openai_provenance import (
    HttpResponse,
    OpenAIContentProvenanceAdapter,
)
from image_trust.watermark.remote import (
    load_remote_verification_settings,
    remote_verification_capabilities,
)


def _asset(tmp_path: Path) -> Path:
    path = tmp_path / "asset.png"
    Image.new("RGB", (32, 32), (20, 40, 60)).save(path)
    return path


def _response(results: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {"created_at": 1, "object": "content_provenance_check", "results": results}
    ).encode("utf-8")


def test_missing_key_never_calls_transport_or_sends_data(tmp_path: Path) -> None:
    called = False

    def transport(_: Request, __: float) -> HttpResponse:
        nonlocal called
        called = True
        raise AssertionError("transport must not run")

    result = OpenAIContentProvenanceAdapter(transport=transport)(_asset(tmp_path))

    assert called is False
    assert result.run_status == "unavailable"
    assert result.observation == "not_observed"
    assert result.data_sent is False
    assert result.limitations == ["openai_api_key_not_configured"]


def test_detected_synthid_is_strong_verified_provider_evidence(tmp_path: Path) -> None:
    def transport(request: Request, timeout: float) -> HttpResponse:
        assert request.full_url.endswith("/v1/content_provenance_checks")
        assert request.get_header("Authorization") == "Bearer test-secret"
        assert timeout == 15.0
        assert b'name="file"' in request.data
        return HttpResponse(
            200,
            _response(
                [
                    {
                        "type": "synthid",
                        "outcome": "detected",
                        "model": "openai-image-test",
                        "generated_at": None,
                    }
                ]
            ),
        )

    result = OpenAIContentProvenanceAdapter(
        api_key="test-secret",
        transport=transport,
    )(_asset(tmp_path))

    assert result.run_status == "ok"
    assert result.observation == "positive"
    assert result.evidence_class == "verified_provider_ai"
    assert result.direction == "supports_ai"
    assert result.strength == "strong"
    assert result.decision_eligible is True
    assert result.data_sent is True
    assert result.provider_signals[0].signal_type == "synthid"
    assert "test-secret" not in result.model_dump_json()


def test_valid_c2pa_is_verified_but_invalid_c2pa_remains_neutral(tmp_path: Path) -> None:
    asset = _asset(tmp_path)

    def valid(_: Request, __: float) -> HttpResponse:
        return HttpResponse(
            200,
            _response(
                [
                    {
                        "type": "c2pa",
                        "outcome": "detected",
                        "validation_state": "valid",
                        "issuer": "OpenAI",
                    }
                ]
            ),
        )

    def invalid(_: Request, __: float) -> HttpResponse:
        return HttpResponse(
            200,
            _response(
                [
                    {
                        "type": "c2pa",
                        "outcome": "detected",
                        "validation_state": "invalid",
                    }
                ]
            ),
        )

    verified = OpenAIContentProvenanceAdapter(api_key="key", transport=valid)(asset)
    unverified = OpenAIContentProvenanceAdapter(api_key="key", transport=invalid)(asset)

    assert verified.decision_eligible is True
    assert unverified.observation == "positive"
    assert unverified.evidence_class == "unverified_identifier"
    assert unverified.direction == "neutral"
    assert unverified.decision_eligible is False


def test_not_detected_and_transport_failures_never_support_camera(tmp_path: Path) -> None:
    asset = _asset(tmp_path)

    def negative(_: Request, __: float) -> HttpResponse:
        return HttpResponse(
            200,
            _response(
                [
                    {"type": "synthid", "outcome": "not_detected"},
                    {
                        "type": "c2pa",
                        "outcome": "not_detected",
                        "validation_state": "not_present",
                    },
                ]
            ),
        )

    def timeout(_: Request, __: float) -> HttpResponse:
        raise TimeoutError

    not_detected = OpenAIContentProvenanceAdapter(api_key="key", transport=negative)(asset)
    timed_out = OpenAIContentProvenanceAdapter(api_key="key", transport=timeout)(asset)

    assert not_detected.observation == "negative"
    assert not_detected.direction == "neutral"
    assert not_detected.decision_eligible is False
    assert "openai_not_detected_does_not_rule_out_ai" in not_detected.limitations
    assert timed_out.run_status == "unavailable"
    assert timed_out.observation == "not_observed"
    assert timed_out.data_sent is True


def test_http_and_malformed_responses_are_sanitized(tmp_path: Path) -> None:
    asset = _asset(tmp_path)

    def unauthorized(_: Request, __: float) -> HttpResponse:
        return HttpResponse(401, b'{"error":{"message":"secret server detail"}}')

    def malformed(_: Request, __: float) -> HttpResponse:
        return HttpResponse(200, b'{"object":"wrong","results":[]}')

    rejected = OpenAIContentProvenanceAdapter(api_key="top-secret", transport=unauthorized)(asset)
    invalid = OpenAIContentProvenanceAdapter(api_key="top-secret", transport=malformed)(asset)

    assert rejected.run_status == "unavailable"
    assert rejected.errors == [
        {"code": "http_401", "message": "OpenAI provenance API returned HTTP 401."}
    ]
    assert invalid.run_status == "failed"
    assert "secret server detail" not in rejected.model_dump_json()
    assert "top-secret" not in rejected.model_dump_json()


def test_settings_load_env_without_exposing_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_SYNTHID_MANUAL_ENABLED", raising=False)
    (tmp_path / ".env").write_text(
        "# local only\nOPENAI_API_KEY='file-secret'\nGOOGLE_SYNTHID_MANUAL_ENABLED=yes\n",
        encoding="utf-8",
    )

    settings = load_remote_verification_settings(tmp_path)
    capabilities = remote_verification_capabilities(settings)

    assert settings.openai_api_key == "file-secret"
    assert "file-secret" not in repr(settings)
    assert capabilities["openai"]["configured"] is True
    assert capabilities["google"]["mode"] == "manual_only"
    assert capabilities["google"]["configured"] is True
    assert "file-secret" not in json.dumps(capabilities)


def test_environment_key_takes_precedence_over_dotenv(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    settings = load_remote_verification_settings(tmp_path)

    assert settings.openai_api_key == "environment-secret"


def test_remote_options_are_hidden_without_environment_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_SYNTHID_MANUAL_ENABLED", raising=False)

    settings = load_remote_verification_settings(tmp_path)
    capabilities = remote_verification_capabilities(settings)

    assert capabilities["openai"]["configured"] is False
    assert capabilities["google"]["configured"] is False
