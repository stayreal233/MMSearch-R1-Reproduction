#!/usr/bin/env python3
"""Run the approved Step-10 512-candidate expansion with v1 replay checks.

This entry point deliberately reuses the audited v1 selector implementation.
Before any model load it binds the immutable v1 negative-result manifest and
the user-approved v2 protocol.  The first 256 replayed records must match the
v1 evidence after removing timing and post-selection annotation fields.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import select_step10_cases as base


V2_SCHEMA = "mmsearch.step10.case-selection.v2"
V2_LIMIT = 512
V1_LIMIT = 256
V2_PROTOCOL_SHA256 = "421cff4b55bb6e0037935b8b334bb076acded571ad89ffd7a70907cd8330ecd4"
V1_PROTOCOL_SHA256 = "63a3ab7a753fb88ac1b923693f8e013052b14ae3e2b960dccf7f6a82095727bc"
V1_MANIFEST_SHA256 = "13fa389ae498de169834896bb61c0f8d8b392b2a626e25981582d048a1f425e7"
V1_MANIFEST_BYTES = 986826
EXPECTED_V2_PROTOCOL = (
    REPO_ROOT / "reproduction/env/step10_selection_protocol_v2.json"
).resolve()
EXPECTED_V1_MANIFEST = Path(
    "/root/autodl-tmp/mmsearch_step10/step10_candidate_selection.json"
)
EXPECTED_V2_OUTPUT_DIR = Path("/root/autodl-tmp/mmsearch_step10_v2/selection")
EXPECTED_V2_MANIFEST = Path(
    "/root/autodl-tmp/mmsearch_step10_v2/step10_candidate_selection_v2.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json_bytes(path: Path, *, maximum: int) -> tuple[dict[str, Any], bytes]:
    resolved = path.resolve(strict=True)
    require(resolved.is_file() and not path.is_symlink(), f"unsafe JSON path: {path}")
    encoded = resolved.read_bytes()
    require(0 < len(encoded) <= maximum, f"unsafe JSON size: {path}")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON: {path}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value, encoded


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def route_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "answer": 0,
        "image_search_text_search_answer": 0,
        "image_search_answer": 0,
        "text_search_answer": 0,
        "other": 0,
    }
    mapping = {
        ("answer",): "answer",
        ("image_search", "text_search", "answer"):
            "image_search_text_search_answer",
        ("image_search", "answer"): "image_search_answer",
        ("text_search", "answer"): "text_search_answer",
    }
    for record in records:
        key = mapping.get(tuple(record.get("action_sequence", [])), "other")
        counts[key] += 1
    return counts


def strict_em(record: dict[str, Any]) -> bool:
    prediction = record.get("final_answer")
    ground_truth = record.get("ground_truth")
    return (
        record.get("terminal_status") == "answered"
        and isinstance(prediction, str)
        and isinstance(ground_truth, str)
        and prediction.strip().lower() == ground_truth.strip().lower()
    )


def replay_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Remove only nondeterministic timings and v2-dependent annotations."""
    projected = copy.deepcopy(record)
    projected.pop("selection_evaluation", None)
    projected.pop("not_selected_reasons", None)
    rounds = projected.get("rounds")
    if isinstance(rounds, list):
        for round_record in rounds:
            if isinstance(round_record, dict):
                round_record.pop("generation_seconds", None)
    return projected


def validate_v1_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    require(path.resolve(strict=True) == EXPECTED_V1_MANIFEST.resolve(strict=True), "v1 manifest path mismatch")
    manifest, encoded = read_json_bytes(path, maximum=4 * 1024 * 1024)
    require(len(encoded) == V1_MANIFEST_BYTES, "v1 manifest byte count changed")
    require(sha256_bytes(encoded) == V1_MANIFEST_SHA256, "v1 manifest digest changed")
    require(manifest.get("schema_version") == "mmsearch.step10.case-selection.v1", "v1 schema mismatch")
    require(manifest.get("status") == "failed_selection_not_found_within_limit", "v1 status mismatch")
    require(manifest.get("scanned_search_required_count") == V1_LIMIT, "v1 count mismatch")
    require(manifest.get("missing_selections") == ["case_b", "case_c"], "v1 missing slots changed")
    records = manifest.get("scan_records")
    require(isinstance(records, list) and len(records) == V1_LIMIT, "v1 records missing")
    require(
        route_counts(records)
        == {
            "answer": 78,
            "image_search_text_search_answer": 178,
            "image_search_answer": 0,
            "text_search_answer": 0,
            "other": 0,
        },
        "v1 route counts changed",
    )
    for number, record in enumerate(records, 1):
        require(record.get("candidate_number") == number, "v1 candidate order mismatch")
        require(record.get("exact_match") is strict_em(record), "v1 strict EM mismatch")
    return manifest, encoded


def validate_v2_protocol(path: Path, v1_manifest: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    require(path.resolve(strict=True) == EXPECTED_V2_PROTOCOL, "v2 protocol path mismatch")
    protocol, encoded = read_json_bytes(path, maximum=1024 * 1024)
    require(sha256_bytes(encoded) == V2_PROTOCOL_SHA256, "v2 protocol digest mismatch")
    require(protocol.get("schema_version") == 2, "v2 protocol schema mismatch")
    require(protocol.get("status") == "registered_before_step10_selection_v2", "v2 protocol status mismatch")
    dataset = protocol.get("dataset")
    require(isinstance(dataset, dict), "v2 dataset protocol missing")
    require(dataset.get("maximum_search_required_candidates") == V2_LIMIT, "v2 limit mismatch")
    require(dataset.get("parquet") == str(base.EXPECTED_PARQUET), "v2 parquet path mismatch")
    model = protocol.get("model")
    require(isinstance(model, dict), "v2 model protocol missing")
    require(model.get("seed") == 0 and model.get("do_sample") is False, "v2 decoding mismatch")
    require(protocol.get("exact_match_definition") == "prediction.strip().lower() == ground_truth.strip().lower()", "v2 EM definition mismatch")
    supersedes = protocol.get("supersedes")
    require(isinstance(supersedes, dict), "v2 parent evidence missing")
    require(supersedes.get("protocol_v1", {}).get("sha256") == V1_PROTOCOL_SHA256, "v1 protocol binding mismatch")
    prior = supersedes.get("failed_selection_v1")
    require(isinstance(prior, dict), "v1 failure binding missing")
    require(prior.get("sha256") == V1_MANIFEST_SHA256 and prior.get("bytes") == V1_MANIFEST_BYTES, "v1 manifest binding mismatch")
    require(prior.get("scanned_search_required_count") == len(v1_manifest["scan_records"]), "v1 bound count mismatch")
    outputs = protocol.get("outputs")
    require(isinstance(outputs, dict), "v2 outputs missing")
    require(outputs.get("selector_output_dir") == str(EXPECTED_V2_OUTPUT_DIR), "v2 output-dir contract mismatch")
    require(outputs.get("selector_manifest") == str(EXPECTED_V2_MANIFEST), "v2 manifest contract mismatch")
    require(outputs.get("overwrite_v1_allowed") is False, "v1 overwrite must be forbidden")
    return protocol, encoded


def replay_check(
    records: list[dict[str, Any]],
    v1_records: list[dict[str, Any]],
) -> dict[str, Any]:
    compared = min(len(records), len(v1_records))
    mismatch_candidates: list[int] = []
    for index in range(compared):
        if replay_projection(records[index]) != replay_projection(v1_records[index]):
            mismatch_candidates.append(index + 1)
    complete = len(records) >= V1_LIMIT
    return {
        "required_candidates": V1_LIMIT,
        "compared_candidates": compared,
        "complete": complete,
        "mismatch_count": len(mismatch_candidates),
        "mismatch_candidates": mismatch_candidates[:20],
        "v1_projection_sha256": canonical_json_sha256(
            [replay_projection(record) for record in v1_records]
        ),
        "v2_prefix_projection_sha256": canonical_json_sha256(
            [replay_projection(record) for record in records[:V1_LIMIT]]
        ) if complete else None,
        "pass": complete and not mismatch_candidates,
    }


def install_v2_contract(
    *,
    protocol_path: Path,
    protocol_bytes: bytes,
    prior_path: Path,
    prior_bytes: bytes,
    prior_manifest: dict[str, Any],
) -> None:
    original_build_manifest = base.build_manifest

    def build_manifest_v2(**kwargs: Any) -> dict[str, Any]:
        payload = original_build_manifest(**kwargs)
        records = kwargs.get("records")
        require(isinstance(records, list), "selector records missing during manifest build")
        replay = replay_check(records, prior_manifest["scan_records"])
        payload["schema_version"] = V2_SCHEMA
        payload["expansion_v2"] = {
            "protocol": {
                "path": str(protocol_path),
                "bytes": len(protocol_bytes),
                "sha256": sha256_bytes(protocol_bytes),
            },
            "prior_v1_manifest": {
                "path": str(prior_path),
                "bytes": len(prior_bytes),
                "sha256": sha256_bytes(prior_bytes),
            },
            "approved_limit": V2_LIMIT,
            "rerun_from_candidate_one": True,
            "v1_replay": replay,
        }
        checks = payload.get("checks")
        require(isinstance(checks, dict), "selector checks missing")
        checks["v1_replay_core_match"] = replay["pass"]
        return payload

    base.MAX_SEARCH_REQUIRED_CANDIDATES = V2_LIMIT
    base.SCHEMA_VERSION = V2_SCHEMA
    base.build_manifest = build_manifest_v2


def verify_written_manifest(path: Path) -> None:
    manifest, _ = read_json_bytes(path, maximum=8 * 1024 * 1024)
    require(manifest.get("schema_version") == V2_SCHEMA, "written v2 schema mismatch")
    require(manifest.get("protocol", {}).get("maximum_search_required_candidates") == V2_LIMIT, "written v2 limit mismatch")
    expansion = manifest.get("expansion_v2")
    require(isinstance(expansion, dict), "written expansion evidence missing")
    replay = expansion.get("v1_replay")
    require(isinstance(replay, dict) and replay.get("pass") is True, "v1 deterministic replay failed")
    require(manifest.get("checks", {}).get("v1_replay_core_match") is True, "v1 replay check not committed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--protocol-v2", type=Path)
    parser.add_argument("--prior-v1-manifest", type=Path)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def run_self_tests() -> None:
    sample = {
        "candidate_number": 1,
        "terminal_status": "answered",
        "final_answer": " A  B ",
        "ground_truth": "a b",
        "exact_match": False,
        "selection_evaluation": {"case_b": {"selected": False}},
        "rounds": [{"action": "answer", "generation_seconds": 1.2}],
    }
    require(strict_em(sample) is False, "strict EM whitespace test failed")
    other = copy.deepcopy(sample)
    other["rounds"][0]["generation_seconds"] = 9.9
    other["selection_evaluation"] = {"case_b": {"selected": True}}
    require(replay_projection(sample) == replay_projection(other), "replay projection test failed")
    counts = route_counts([
        {"action_sequence": ["answer"]},
        {"action_sequence": ["image_search", "text_search", "answer"]},
    ])
    require(counts["answer"] == 1 and counts["image_search_text_search_answer"] == 1, "route count test failed")
    print(json.dumps({"status": "passed", "pure_self_tests": 3}, indent=2))


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return 0
    for name in (
        "protocol_v2",
        "prior_v1_manifest",
        "parquet",
        "model_path",
        "output_dir",
        "manifest",
    ):
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    require(args.output_dir.resolve() == EXPECTED_V2_OUTPUT_DIR, "v2 output directory mismatch")
    require(args.manifest.resolve() == EXPECTED_V2_MANIFEST, "v2 manifest path mismatch")
    require(not args.manifest.exists(), "v2 manifest already exists")

    prior_manifest, prior_bytes = validate_v1_manifest(args.prior_v1_manifest)
    _, protocol_bytes = validate_v2_protocol(args.protocol_v2, prior_manifest)
    if args.validate_only:
        print(json.dumps({
            "status": "passed",
            "mode": "validate_only",
            "protocol_v2_sha256": sha256_bytes(protocol_bytes),
            "prior_v1_manifest_sha256": sha256_bytes(prior_bytes),
            "approved_limit": V2_LIMIT,
        }, indent=2))
        return 0
    install_v2_contract(
        protocol_path=args.protocol_v2.resolve(strict=True),
        protocol_bytes=protocol_bytes,
        prior_path=args.prior_v1_manifest.resolve(strict=True),
        prior_bytes=prior_bytes,
        prior_manifest=prior_manifest,
    )

    original_argv = sys.argv
    sys.argv = [
        str(base.SCRIPT_PATH),
        "--parquet", str(args.parquet),
        "--model-path", str(args.model_path),
        "--output-dir", str(args.output_dir),
        "--manifest", str(args.manifest),
    ]
    exit_code = 0
    pending_error: BaseException | None = None
    try:
        base.main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except BaseException as exc:  # noqa: BLE001 - preserve base failure after audit
        pending_error = exc
    finally:
        sys.argv = original_argv

    if args.manifest.exists():
        verify_written_manifest(args.manifest)
    if pending_error is not None:
        raise pending_error
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
