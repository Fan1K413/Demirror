from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from image_trust.provenance.c2pa import inspect_c2pa_asset, write_c2pa_record
from image_trust.provenance.contracts import (
    C2paRecord,
    C2paRecordStatus,
    C2paSignatureValidationStatus,
    C2paTrustStatus,
    P1C2paConfig,
)


class FakeContext:
    settings = None

    @classmethod
    def from_dict(cls, settings):
        cls.settings = settings
        return cls()

    def close(self):
        return None


class FakeReader:
    result = None

    @classmethod
    def try_create(cls, path, *, context):
        assert path.name == "asset.jpg"
        assert context is not None
        return cls.result


class ManifestReader:
    def json(self):
        return json.dumps({"active_manifest": "urn:c2pa:example"})

    def get_active_manifest(self):
        return {
            "assertions": [
                {
                    "label": "c2pa.actions.v2",
                    "data": {
                        "actions": [
                            {"action": "c2pa.opened"},
                            {"action": "c2pa.created"},
                        ],
                        "digitalSourceType": "trainedAlgorithmicMedia",
                    },
                },
                {"label": "c2pa.hash.data", "data": {}},
            ]
        }

    def get_validation_state(self):
        return "valid"

    def get_validation_results(self):
        return {"validation_errors": [{"code": "example.warning"}]}

    def close(self):
        return None


class UntrustedManifestReader(ManifestReader):
    def get_validation_results(self):
        return {"validation_errors": [{"code": "signingCredential.untrusted"}]}


class FakeC2paModule:
    __version__ = "0.32.6"
    Context = FakeContext
    Reader = FakeReader


def test_no_manifest_is_not_observed_and_reader_is_offline(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "fake_c2pa", FakeC2paModule)
    FakeReader.result = None

    record = inspect_c2pa_asset(_asset(tmp_path), _config())

    assert record.status.value == "not_observed"
    assert record.manifest_present is False
    assert record.signature_validation_status.value == "not_observed"
    assert record.config_version == "test"
    assert record.input_sha256 is not None
    assert len(record.input_sha256) == 64
    assert FakeContext.settings["verify"] == {
        "remote_manifest_fetch": False,
        "ocsp_fetch": False,
    }
    assert "offline_validation_cannot_confirm_current_revocation_state" in record.limitations


def test_manifest_claims_are_recorded_without_interpreting_them(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "fake_c2pa", FakeC2paModule)
    FakeReader.result = ManifestReader()
    config = _config(trust_list_version="offline-trust-list-2026-08-08")

    record = inspect_c2pa_asset(_asset(tmp_path), config)
    output = tmp_path / "c2pa_result.json"
    write_c2pa_record(output, record)

    assert record.status.value == "present"
    assert record.assertion_labels == ["c2pa.actions.v2", "c2pa.hash.data"]
    assert record.declared_actions == ["c2pa.created", "c2pa.opened"]
    assert record.declared_digital_source_types == ["trainedAlgorithmicMedia"]
    assert record.signature_validation_status.value == "valid"
    assert record.trust_status.value == "trusted"
    assert json.loads(output.read_text(encoding="utf-8"))["network_access"] == "disabled"


def test_missing_c2pa_dependency_is_explicitly_unavailable(tmp_path) -> None:
    record = inspect_c2pa_asset(
        _asset(tmp_path),
        P1C2paConfig(
            config_version="test",
            module_name="demirror_missing_c2pa",
            expected_dependency_version="0.32.6",
        ),
    )

    assert record.status.value == "unavailable"
    assert "c2pa_python_not_installed" in record.limitations


def test_untrusted_validation_and_missing_input_are_explicit(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "fake_c2pa", FakeC2paModule)
    FakeReader.result = UntrustedManifestReader()
    record = inspect_c2pa_asset(
        _asset(tmp_path),
        _config(trust_list_version="offline-trust-list-2026-08-08"),
    )
    missing = inspect_c2pa_asset(tmp_path / "missing.jpg", _config())

    assert record.trust_status.value == "untrusted"
    assert missing.status.value == "failed"
    assert "input_path_is_not_a_file" in missing.limitations


def test_c2pa_contract_rejects_a_present_status_without_a_manifest() -> None:
    with pytest.raises(ValueError, match="require a manifest"):
        C2paRecord(
            config_version="test",
            config_digest="0" * 64,
            original_filename="asset.jpg",
            status=C2paRecordStatus.PRESENT,
            manifest_present=False,
            signature_validation_status=C2paSignatureValidationStatus.INDETERMINATE,
            trust_status=C2paTrustStatus.INDETERMINATE,
            trust_list_version="not_configured",
        )


def _config(*, trust_list_version: str = "not_configured") -> P1C2paConfig:
    return P1C2paConfig(
        config_version="test",
        module_name="fake_c2pa",
        expected_dependency_version="0.32.6",
        trust_list_version=trust_list_version,
    )


def _asset(tmp_path) -> Path:
    path = tmp_path / "asset.jpg"
    path.write_bytes(b"not-a-real-jpeg")
    return path
