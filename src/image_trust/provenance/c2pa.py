"""Offline-only C2PA manifest inspection for local assets."""

from __future__ import annotations

import importlib
import json
import mimetypes
from hashlib import sha256
from pathlib import Path
from typing import Any

from image_trust.provenance.contracts import (
    C2paRecord,
    C2paRecordStatus,
    C2paSignatureValidationStatus,
    C2paTrustStatus,
    P1C2paConfig,
)


def inspect_c2pa_asset(input_path: Path, config: P1C2paConfig) -> C2paRecord:
    """Read and validate one local C2PA store without network access.

    Remote manifests and OCSP are both disabled before creating the SDK
    reader.  Therefore a record is limited to embedded local information and
    cannot claim that revocation information is current.
    """

    media_type = _asset_media_type(input_path)
    if not input_path.is_file():
        return _error_record(
            media_type,
            config,
            sdk_version=None,
            error=FileNotFoundError(input_path),
            limitation="input_path_is_not_a_file",
            status=C2paRecordStatus.FAILED,
            input_path=input_path,
        )
    try:
        input_sha256 = _sha256(input_path)
    except OSError as error:
        return _error_record(
            media_type,
            config,
            sdk_version=None,
            error=error,
            limitation="input_hash_unavailable",
            status=C2paRecordStatus.FAILED,
            input_path=input_path,
        )
    try:
        module = importlib.import_module(config.module_name)
    except ModuleNotFoundError:
        return _unavailable_record(
            media_type,
            config,
            "c2pa_python_not_installed",
            input_path=input_path,
            input_sha256=input_sha256,
        )
    version = getattr(module, "__version__", None)
    if version != config.expected_dependency_version:
        return _unavailable_record(
            media_type,
            config,
            "c2pa_python_version_does_not_match_config",
            sdk_version=str(version) if version is not None else None,
            input_path=input_path,
            input_sha256=input_sha256,
        )
    try:
        context = module.Context.from_dict(
            {
                "verify": {
                    "remote_manifest_fetch": False,
                    "ocsp_fetch": False,
                }
            }
        )
        reader = module.Reader.try_create(input_path, context=context)
    except Exception as error:  # SDK errors are serialized into an audit record.
        return _error_record(
            media_type,
            config,
            str(version),
            error,
            input_path=input_path,
            input_sha256=input_sha256,
        )
    finally:
        _close_safely(locals().get("context"))
    if reader is None:
        return C2paRecord(
            **_record_context(config, input_path, input_sha256),
            status=C2paRecordStatus.NOT_OBSERVED,
            manifest_present=False,
            asset_media_type=media_type,
            signature_validation_status=C2paSignatureValidationStatus.NOT_OBSERVED,
            trust_status=C2paTrustStatus.NOT_ASSESSED,
            trust_list_version=config.trust_list_version,
            sdk_version=str(version),
            limitations=_offline_limitations(config, "no_embedded_c2pa_manifest_found"),
        )
    try:
        manifest_store = json.loads(reader.json())
        active_manifest = reader.get_active_manifest()
        validation_state = reader.get_validation_state()
        validation_results = reader.get_validation_results()
    except Exception as error:  # A readable store with invalid structure is malformed.
        return _error_record(
            media_type,
            config,
            str(version),
            error,
            manifest_present=True,
            input_path=input_path,
            input_sha256=input_sha256,
        )
    finally:
        _close_safely(reader)
    if not isinstance(active_manifest, dict):
        return C2paRecord(
            **_record_context(config, input_path, input_sha256),
            status=C2paRecordStatus.MALFORMED,
            manifest_present=True,
            asset_media_type=media_type,
            validation_state=_as_text(validation_state),
            validation_status_codes=_validation_codes(validation_results),
            signature_validation_status=C2paSignatureValidationStatus.INDETERMINATE,
            trust_status=C2paTrustStatus.INDETERMINATE,
            trust_list_version=config.trust_list_version,
            sdk_version=str(version),
            limitations=_offline_limitations(config, "active_manifest_not_available"),
        )
    validation_codes = _validation_codes(validation_results)
    signature_status = _signature_status(validation_state)
    return C2paRecord(
        **_record_context(config, input_path, input_sha256),
        status=C2paRecordStatus.PRESENT,
        manifest_present=True,
        asset_media_type=media_type,
        active_manifest_label=_active_manifest_label(manifest_store, active_manifest),
        assertion_labels=_assertion_labels(active_manifest),
        declared_actions=_declared_actions(active_manifest),
        declared_digital_source_types=_digital_source_types(active_manifest),
        validation_state=_as_text(validation_state),
        validation_status_codes=validation_codes,
        signature_validation_status=signature_status,
        trust_status=_trust_status(config, signature_status, validation_codes),
        trust_list_version=config.trust_list_version,
        sdk_version=str(version),
        limitations=_offline_limitations(
            config,
            "signature_status_is_mapped_from_sdk_validation_state",
            "claims_are_recorded_without_interpreting_real_world_events",
        ),
    )


def write_c2pa_record(path: Path, record: C2paRecord) -> None:
    """Write one C2PA audit record atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _unavailable_record(
    media_type: str | None,
    config: P1C2paConfig,
    limitation: str,
    *,
    sdk_version: str | None = None,
    input_path: Path,
    input_sha256: str,
) -> C2paRecord:
    return C2paRecord(
        **_record_context(config, input_path, input_sha256),
        status=C2paRecordStatus.UNAVAILABLE,
        manifest_present=False,
        asset_media_type=media_type,
        signature_validation_status=C2paSignatureValidationStatus.NOT_OBSERVED,
        trust_status=C2paTrustStatus.NOT_ASSESSED,
        trust_list_version=config.trust_list_version,
        sdk_version=sdk_version,
        limitations=_offline_limitations(config, limitation),
    )


def _error_record(
    media_type: str | None,
    config: P1C2paConfig,
    sdk_version: str | None,
    error: Exception,
    *,
    manifest_present: bool = False,
    limitation: str | None = None,
    status: C2paRecordStatus | None = None,
    input_path: Path,
    input_sha256: str | None = None,
) -> C2paRecord:
    status = status or (
        C2paRecordStatus.MALFORMED
        if manifest_present
        else C2paRecordStatus.FAILED
    )
    return C2paRecord(
        **_record_context(config, input_path, input_sha256),
        status=status,
        manifest_present=manifest_present,
        asset_media_type=media_type,
        signature_validation_status=C2paSignatureValidationStatus.INDETERMINATE,
        trust_status=C2paTrustStatus.INDETERMINATE,
        trust_list_version=config.trust_list_version,
        sdk_version=sdk_version,
        limitations=_offline_limitations(
            config,
            limitation or f"c2pa_sdk_error:{type(error).__name__}",
        ),
    )


def _offline_limitations(config: P1C2paConfig, *additional: str) -> list[str]:
    limitations = [
        "remote_manifest_fetch_disabled",
        "ocsp_fetch_disabled",
        "offline_validation_cannot_confirm_current_revocation_state",
        *additional,
    ]
    if config.trust_list_version == "not_configured":
        limitations.append("trust_list_version_not_configured")
    return sorted(set(limitations))


def _record_context(
    config: P1C2paConfig,
    input_path: Path,
    input_sha256: str | None,
) -> dict[str, str | None]:
    return {
        "config_version": config.config_version,
        "config_digest": _config_digest(config),
        "input_sha256": input_sha256,
        "original_filename": input_path.name,
    }


def _config_digest(config: P1C2paConfig) -> str:
    payload = config.model_dump_json(exclude_none=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_media_type(path: Path) -> str | None:
    known = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return known.get(path.suffix.lower(), mimetypes.guess_type(path.name)[0])


def _assertion_labels(active_manifest: dict[str, Any]) -> list[str]:
    assertions = active_manifest.get("assertions", [])
    if isinstance(assertions, dict):
        return sorted(str(label) for label in assertions)
    if not isinstance(assertions, list):
        return []
    labels: list[str] = []
    for assertion in assertions:
        if isinstance(assertion, dict):
            label = assertion.get("label")
            if isinstance(label, str):
                labels.append(label)
    return sorted(set(labels))


def _declared_actions(active_manifest: dict[str, Any]) -> list[str]:
    actions: set[str] = set()
    for label, payload in _iter_assertions(active_manifest):
        if "actions" not in label.lower():
            continue
        _collect_actions(payload, actions)
    return sorted(actions)


def _digital_source_types(active_manifest: dict[str, Any]) -> list[str]:
    """Record declared C2PA source types without treating them as ground truth."""

    source_types: set[str] = set()
    _collect_digital_source_types(active_manifest, source_types)
    return sorted(source_types)


def _collect_digital_source_types(value: Any, source_types: set[str]) -> None:
    if isinstance(value, dict):
        for key in ("digitalSourceType", "digital_source_type"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                source_types.add(candidate.strip())
        for nested in value.values():
            _collect_digital_source_types(nested, source_types)
    elif isinstance(value, list):
        for nested in value:
            _collect_digital_source_types(nested, source_types)


def _iter_assertions(active_manifest: dict[str, Any]):
    assertions = active_manifest.get("assertions", [])
    if isinstance(assertions, dict):
        yield from assertions.items()
    elif isinstance(assertions, list):
        for assertion in assertions:
            if isinstance(assertion, dict) and isinstance(assertion.get("label"), str):
                yield assertion["label"], assertion.get("data", assertion)


def _collect_actions(value: Any, actions: set[str]) -> None:
    if isinstance(value, dict):
        action_name = value.get("action")
        if isinstance(action_name, str):
            actions.add(action_name)
        for nested in value.values():
            _collect_actions(nested, actions)
    elif isinstance(value, list):
        for nested in value:
            _collect_actions(nested, actions)


def _validation_codes(value: Any) -> list[str]:
    codes: set[str] = set()
    _collect_validation_codes(value, codes)
    return sorted(codes)


def _collect_validation_codes(value: Any, codes: set[str]) -> None:
    if isinstance(value, dict):
        code = value.get("code")
        if isinstance(code, str):
            codes.add(code)
        for nested in value.values():
            _collect_validation_codes(nested, codes)
    elif isinstance(value, list):
        for nested in value:
            _collect_validation_codes(nested, codes)


def _signature_status(value: Any) -> C2paSignatureValidationStatus:
    normalized = _as_text(value)
    if normalized is None:
        return C2paSignatureValidationStatus.INDETERMINATE
    text = normalized.lower()
    if "invalid" in text or "fail" in text:
        return C2paSignatureValidationStatus.INVALID
    if text in {"valid", "validated"}:
        return C2paSignatureValidationStatus.VALID
    return C2paSignatureValidationStatus.INDETERMINATE


def _trust_status(
    config: P1C2paConfig,
    signature_status: C2paSignatureValidationStatus,
    validation_codes: list[str],
) -> C2paTrustStatus:
    if config.trust_list_version == "not_configured":
        return C2paTrustStatus.NOT_ASSESSED
    if any("untrusted" in code.lower() for code in validation_codes):
        return C2paTrustStatus.UNTRUSTED
    if signature_status is C2paSignatureValidationStatus.VALID:
        return C2paTrustStatus.TRUSTED
    return C2paTrustStatus.INDETERMINATE


def _active_manifest_label(manifest_store: dict[str, Any], active_manifest: dict[str, Any]) -> str | None:
    label = active_manifest.get("label")
    if isinstance(label, str):
        return label
    label = manifest_store.get("active_manifest")
    return label if isinstance(label, str) else None


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _close_safely(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()
