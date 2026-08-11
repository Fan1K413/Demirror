"""Opt-in adapter for OpenAI's official content-provenance endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import mimetypes
from pathlib import Path
import socket
from typing import Callable, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from image_trust.watermark.contracts import (
    WatermarkAdapterResult,
    WatermarkCoverage,
    WatermarkProviderSignal,
)
from image_trust.watermark.remote import OPENAI_PROVENANCE_ENDPOINT


MAX_RESPONSE_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 25 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 15.0


class HttpResponse(NamedTuple):
    status: int
    body: bytes


Transport = Callable[[Request, float], HttpResponse]


def _default_transport(request: Request, timeout: float) -> HttpResponse:
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed official endpoint
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("openai_response_too_large")
        return HttpResponse(status=int(response.status), body=body)


@dataclass(frozen=True)
class OpenAIContentProvenanceAdapter:
    """Check only supported OpenAI C2PA and SynthID provenance signals."""

    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    endpoint: str = OPENAI_PROVENANCE_ENDPOINT
    transport: Transport = field(default=_default_transport, repr=False, compare=False)

    adapter_id = "openai_content_provenance_api_v1"
    scheme = "openai_c2pa_synthid"
    detector_version = "openai-content-provenance-api-v1"
    coverage = WatermarkCoverage(
        ecosystem=["OpenAI"],
        supported_formats=["jpeg", "png", "webp"],
    )

    def __call__(self, input_path: Path) -> WatermarkAdapterResult:
        common = {
            "adapter_id": self.adapter_id,
            "scheme": self.scheme,
            "detector_version": self.detector_version,
            "coverage": self.coverage,
            "provider": "openai",
            "network_access": "explicit_opt_in",
        }
        if not self.api_key:
            return WatermarkAdapterResult(
                **common,
                run_status="unavailable",
                observation="not_observed",
                limitations=["openai_api_key_not_configured"],
            )
        try:
            size = input_path.stat().st_size
        except OSError as error:
            return WatermarkAdapterResult(
                **common,
                run_status="failed",
                observation="not_observed",
                limitations=["openai_provenance_input_unavailable"],
                errors=[{"code": type(error).__name__, "message": "Could not read the selected image."}],
            )
        if size > MAX_REQUEST_BYTES:
            return WatermarkAdapterResult(
                **common,
                run_status="not_applicable",
                observation="not_observed",
                limitations=["openai_provenance_input_too_large"],
            )

        try:
            request = self._request(input_path)
        except OSError as error:
            return WatermarkAdapterResult(
                **common,
                run_status="failed",
                observation="not_observed",
                limitations=["openai_provenance_input_unavailable"],
                errors=[{"code": type(error).__name__, "message": "Could not read the selected image."}],
            )

        try:
            response = self.transport(request, self.timeout_seconds)
        except HTTPError as error:
            return self._http_failure(error.code, common)
        except (TimeoutError, socket.timeout):
            return WatermarkAdapterResult(
                **common,
                run_status="unavailable",
                observation="not_observed",
                data_sent=True,
                limitations=["openai_provenance_request_timed_out"],
                errors=[{"code": "timeout", "message": "OpenAI provenance check timed out."}],
            )
        except (URLError, OSError) as error:
            return WatermarkAdapterResult(
                **common,
                run_status="unavailable",
                observation="not_observed",
                data_sent=True,
                limitations=["openai_provenance_network_unavailable"],
                errors=[{"code": type(error).__name__, "message": "OpenAI provenance API was unreachable."}],
            )
        except ValueError as error:
            return WatermarkAdapterResult(
                **common,
                run_status="failed",
                observation="not_observed",
                data_sent=True,
                limitations=[str(error) if str(error).startswith("openai_") else "openai_provenance_response_invalid"],
                errors=[{"code": "invalid_response", "message": "OpenAI provenance response was invalid."}],
            )
        except Exception as error:
            return WatermarkAdapterResult(
                **common,
                run_status="failed",
                observation="not_observed",
                data_sent=True,
                limitations=["openai_provenance_unhandled_failure"],
                errors=[
                    {
                        "code": type(error).__name__,
                        "message": "OpenAI provenance check failed without exposing request details.",
                    }
                ],
            )
        if response.status != 200:
            return self._http_failure(response.status, common)
        return self._parse_response(response.body, common)

    def _request(self, input_path: Path) -> Request:
        boundary = f"demirror-{uuid.uuid4().hex}"
        media_type = mimetypes.guess_type(input_path.name)[0] or "application/octet-stream"
        filename = input_path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode("utf-8")
        body = prefix + input_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
        return Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "Demirror/0.1",
            },
        )

    def _http_failure(self, status: int, common: dict[str, object]) -> WatermarkAdapterResult:
        limitation = {
            401: "openai_api_key_rejected",
            403: "openai_provenance_access_denied",
            429: "openai_provenance_rate_limited",
        }.get(status, "openai_provenance_http_error")
        return WatermarkAdapterResult(
            **common,
            run_status="unavailable" if status in {401, 403, 429} or status >= 500 else "failed",
            observation="not_observed",
            data_sent=True,
            limitations=[limitation],
            errors=[{"code": f"http_{status}", "message": f"OpenAI provenance API returned HTTP {status}."}],
        )

    def _parse_response(
        self,
        body: bytes,
        common: dict[str, object],
    ) -> WatermarkAdapterResult:
        try:
            payload = json.loads(body.decode("utf-8"))
            raw_results = payload["results"]
            if payload.get("object") != "content_provenance_check" or not isinstance(raw_results, list):
                raise ValueError
            signals = [_parse_signal(item) for item in raw_results]
            if not signals:
                raise ValueError
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return WatermarkAdapterResult(
                **common,
                run_status="failed",
                observation="not_observed",
                data_sent=True,
                limitations=["openai_provenance_response_invalid"],
                errors=[{"code": "invalid_response", "message": "OpenAI provenance response was invalid."}],
            )

        verified = any(
            signal.outcome == "detected"
            and (
                signal.signal_type == "synthid"
                or signal.validation_state in {"trusted", "valid"}
            )
            for signal in signals
        )
        unverified = any(signal.outcome == "detected" for signal in signals) and not verified
        limitations = ["openai_provider_signal_only_covers_supported_openai_content"]
        if verified:
            return WatermarkAdapterResult(
                **common,
                run_status="ok",
                observation="positive",
                evidence_class="verified_provider_ai",
                direction="supports_ai",
                strength="strong",
                decision_eligible=True,
                provider_signals=signals,
                data_sent=True,
                limitations=limitations,
            )
        if unverified:
            return WatermarkAdapterResult(
                **common,
                run_status="ok",
                observation="positive",
                evidence_class="unverified_identifier",
                provider_signals=signals,
                data_sent=True,
                limitations=[*limitations, "openai_c2pa_signal_not_validated"],
            )
        return WatermarkAdapterResult(
            **common,
            run_status="ok",
            observation="negative",
            provider_signals=signals,
            data_sent=True,
            limitations=[*limitations, "openai_not_detected_does_not_rule_out_ai"],
        )


def _parse_signal(value: object) -> WatermarkProviderSignal:
    if not isinstance(value, dict):
        raise ValueError("invalid signal")
    signal_type = value.get("type")
    outcome = value.get("outcome")
    if signal_type not in {"c2pa", "synthid"} or outcome not in {"detected", "not_detected"}:
        raise ValueError("invalid signal")
    validation = value.get("validation_state")
    if signal_type == "c2pa" and validation not in {"trusted", "valid", "invalid", "not_present"}:
        raise ValueError("invalid C2PA state")
    if signal_type == "synthid":
        validation = None
    return WatermarkProviderSignal(
        signal_type=signal_type,
        outcome=outcome,
        validation_state=validation,
        model=_optional_text(value.get("model")),
        issuer=_optional_text(value.get("issuer")),
        generated_at=_optional_text(value.get("generated_at")),
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
