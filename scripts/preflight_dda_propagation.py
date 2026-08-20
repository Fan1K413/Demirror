"""Run one resource-bounded DDA propagation preflight.

The product worker is launched through the active Python executable.  The
launcher and every descendant process are sampled throughout model loading and
inference, and the process tree is stopped if their summed working set crosses
the registered limit.  A score is interpreted only when it also matches a
prior label-blind product probe on the same frozen artifact.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> str:
    """Hash Python source paths and bytes in a stable, path-independent order."""

    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"No Python sources found under {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing registered file: {path}")
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def _select_variant(repo_root: Path, protocol: Mapping[str, Any]) -> Path:
    registration = protocol["variant_manifest"]
    manifest_path = repo_root / str(registration["path"])
    _require_hash(manifest_path, str(registration["sha256"]))
    records = _read_json(manifest_path).get("records")
    if not isinstance(records, list) or len(records) != int(registration["record_count"]):
        raise ValueError("Registered variant manifest count mismatch")
    selected = protocol["input"]
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and str(record.get("source_asset_sha256", "")).lower()
        == str(selected["source_asset_sha256"]).lower()
        and str(record.get("profile", "")) == str(selected["profile"])
    ]
    if len(matches) != 1:
        raise ValueError("Registered DDA preflight view is not unique")
    record = matches[0]
    if str(record.get("artifact_sha256", "")).lower() != str(selected["artifact_sha256"]).lower():
        raise ValueError("Registered DDA preflight artifact hash mismatch")
    artifact = (manifest_path.parent / str(record.get("relative_path", ""))).resolve()
    try:
        artifact.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("DDA preflight artifact escapes manifest root") from error
    _require_hash(artifact, str(selected["artifact_sha256"]))
    return artifact


def _prior_score(repo_root: Path, protocol: Mapping[str, Any]) -> float:
    registration = protocol["prior_probe"]
    path = repo_root / str(registration["path"])
    _require_hash(path, str(registration["sha256"]))
    report = _read_json(path)
    selected = protocol["input"]
    if report.get("input") != selected:
        raise ValueError("Prior product probe input differs from DDA preflight input")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Prior product probe rows are missing")
    matches = [row for row in rows if isinstance(row, dict) and row.get("detector") == "dda"]
    if len(matches) != 1 or matches[0].get("status") != "available":
        raise ValueError("Prior product probe has no unique available DDA score")
    return float(matches[0]["score"])


def _working_set_bytes(pid: int) -> int | None:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        process_query_information = 0x0400
        process_vm_read = 0x0010
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
        if not handle:
            return None
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            success = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return int(counters.working_set_size) if success else None
        finally:
            kernel32.CloseHandle(handle)
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _process_parent_map() -> dict[int, int]:
    if os.name == "nt":
        class ProcessEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_ulong),
                ("cntUsage", ctypes.c_ulong),
                ("th32ProcessID", ctypes.c_ulong),
                ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", ctypes.c_ulong),
                ("cntThreads", ctypes.c_ulong),
                ("th32ParentProcessID", ctypes.c_ulong),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
        kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32)]
        kernel32.Process32FirstW.restype = ctypes.c_int
        kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32)]
        kernel32.Process32NextW.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        invalid_handle = ctypes.c_void_p(-1).value
        if snapshot == invalid_handle:
            return {}
        parents: dict[int, int] = {}
        try:
            entry = ProcessEntry32()
            entry.dwSize = ctypes.sizeof(entry)
            success = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while success:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                success = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
        return parents
    parents = {}
    for status in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = status.read_text(encoding="utf-8")
            close = text.rfind(")")
            pid = int(status.parent.name)
            parent = int(text[close + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        parents[pid] = parent
    return parents


def _descendant_pids(root_pid: int, parents: Mapping[int, int]) -> set[int]:
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in descendants and parent in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _process_tree_working_set_bytes(root_pid: int) -> tuple[int, int]:
    pids = _descendant_pids(root_pid, _process_parent_map())
    samples = [sample for pid in pids if (sample := _working_set_bytes(pid)) is not None]
    return sum(samples), len(samples)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    elif process.poll() is None:
        process.kill()


def _warning_codes(stderr: str) -> list[str]:
    codes = []
    if "xFormers is not available" in stderr:
        codes.append("xformers_not_available_cpu_fallback_used")
    return codes


def _run_worker(
    command: list[str],
    environment: Mapping[str, str],
    *,
    maximum_working_set_bytes: int,
    timeout_seconds: float,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(environment),
    )
    peak = 0
    peak_process_count = 0
    stop_reason: str | None = None
    while process.poll() is None:
        sample, process_count = _process_tree_working_set_bytes(process.pid)
        if process_count:
            peak = max(peak, sample)
            peak_process_count = max(peak_process_count, process_count)
            if sample > maximum_working_set_bytes:
                stop_reason = "working_set_limit_exceeded"
                _terminate_process_tree(process)
                break
        if time.monotonic() - started > timeout_seconds:
            stop_reason = "timeout_exceeded"
            _terminate_process_tree(process)
            break
        time.sleep(sample_interval_seconds)
    stdout, stderr = process.communicate()
    return {
        "returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "maximum_sampled_working_set_bytes": peak,
        "maximum_sampled_process_count": peak_process_count,
        "stop_reason": stop_reason,
        "stdout_present": bool(stdout.strip()),
        "warning_codes": _warning_codes(stderr),
        "stderr_present_unclassified": bool(stderr.strip()) and not _warning_codes(stderr),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    implementation = protocol["implementation"]
    script_path = args.repo_root / str(implementation["preflight_script_path"])
    runtime_path = args.repo_root / str(implementation["runtime_adapter_path"])
    _require_hash(script_path, str(implementation["preflight_script_sha256"]))
    _require_hash(runtime_path, str(implementation["runtime_adapter_sha256"]))
    model = protocol["model"]
    checkpoint_path = args.repo_root / str(model["checkpoint_path"])
    _require_hash(checkpoint_path, str(model["checkpoint_sha256"]))
    _require_hash(args.repo_root / str(model["audit_path"]), str(model["audit_sha256"]))
    _require_hash(
        args.repo_root / str(model["low_memory_audit_path"]),
        str(model["low_memory_audit_sha256"]),
    )
    torch_hub_root = Path.home() / ".cache" / "torch" / "hub" / str(implementation["torch_hub_directory"])
    if _python_tree_sha256(torch_hub_root) != str(implementation["torch_hub_python_tree_sha256"]):
        raise ValueError("Registered local DINOv2 Python source tree mismatch")
    input_path = _select_variant(args.repo_root, protocol)
    prior_score = _prior_score(args.repo_root, protocol)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "worker-result.json"
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "DDA_CPU_THREADS": "4",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    execution = protocol["execution"]
    command = [
        sys.executable,
        "-m",
        "image_trust.ai_likelihood.dda",
        "--worker",
        "--input",
        str(input_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(result_path),
    ]
    observation = _run_worker(
        command,
        environment,
        maximum_working_set_bytes=int(execution["maximum_working_set_bytes"]),
        timeout_seconds=float(execution["timeout_seconds"]),
        sample_interval_seconds=float(execution["sample_interval_seconds"]),
    )
    score: float | None = None
    preprocessing: str | None = None
    score_difference: float | None = None
    if observation["returncode"] == 0 and result_path.is_file():
        result = _read_json(result_path)
        score = float(result["score"])
        preprocessing = str(result["preprocessing"])
        score_difference = abs(score - prior_score)
    equivalence_passed = (
        score_difference is not None
        and score_difference <= float(protocol["equivalence"]["absolute_difference_maximum"])
        and preprocessing == str(protocol["equivalence"]["preprocessing"])
    )
    resource_passed = (
        observation["stop_reason"] is None
        and observation["returncode"] == 0
        and 0 < int(observation["maximum_sampled_working_set_bytes"])
        <= int(execution["maximum_working_set_bytes"])
    )
    report = {
        "schema_version": "demirror-dda-propagation-preflight-v2",
        "purpose": "One label-blind resource and runtime-equivalence preflight; no accuracy, threshold, score or web-policy decision is authorized.",
        "protocol_sha256": protocol_hash,
        "input": dict(protocol["input"]),
        "resource": {
            "maximum_working_set_bytes": int(execution["maximum_working_set_bytes"]),
            "maximum_sampled_working_set_bytes": int(observation["maximum_sampled_working_set_bytes"]),
            "maximum_sampled_process_count": int(observation["maximum_sampled_process_count"]),
            "sample_interval_seconds": float(execution["sample_interval_seconds"]),
            "sampling_scope": "launcher_and_all_descendants",
            "elapsed_seconds": float(observation["elapsed_seconds"]),
            "stop_reason": observation["stop_reason"],
            "passed": resource_passed,
        },
        "runtime_equivalence": {
            "prior_product_score": prior_score,
            "preflight_score": score,
            "absolute_difference": score_difference,
            "maximum": float(protocol["equivalence"]["absolute_difference_maximum"]),
            "preprocessing": preprocessing,
            "passed": equivalence_passed,
        },
        "worker_returncode": observation["returncode"],
        "worker_warning_codes": observation["warning_codes"],
        "worker_stderr_present_unclassified": observation["stderr_present_unclassified"],
        "full_propagation_audit_resource_preflight_passed": resource_passed and equivalence_passed,
        "deployment_eligible": False,
        "runtime_policy_changed": False,
    }
    _atomic_write_json(args.audit_output, report)
    return report


def _path(value: str) -> Path:
    return Path(value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=_path, default=Path.cwd())
    parser.add_argument(
        "--protocol",
        type=_path,
        default=Path("research/records/2026-08-20/pixel/dda_propagation_preflight_protocol_v2.json"),
    )
    parser.add_argument(
        "--output-dir", type=_path, default=Path("outputs/research/dda_propagation_preflight_v2")
    )
    parser.add_argument(
        "--audit-output",
        type=_path,
        default=Path("research/records/2026-08-20/pixel/dda_propagation_preflight_v2.json"),
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["full_propagation_audit_resource_preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
