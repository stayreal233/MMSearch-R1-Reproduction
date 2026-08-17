#!/usr/bin/env python3
"""Build the five immutable Step-10 inputs for the approved controlled run.

No model, network, environment credential, or search tool is touched here.
The natural 512-candidate failure manifest remains the source of selection
provenance; B/C are explicitly marked as controller interventions.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import prepare_step10_suite_inputs as bridge


PROTOCOL = Path(
    "/root/autodl-tmp/multimodal-search-r1/"
    "reproduction/env/step10_controlled_route_protocol.json"
)
PROTOCOL_SHA256 = "f18ac1cdb69e035243079c56b5abc4806609ada4d522e19ed0336d744ad5c48f"
NATURAL_MANIFEST = Path(
    "/root/autodl-tmp/mmsearch_step10_v2/step10_candidate_selection_v2.json"
)
NATURAL_MANIFEST_SHA256 = (
    "a512d87f3dd24fc5a2714f58589a846f00d494132e678cc2479fd5410751b17d"
)
PARQUET = Path("/root/autodl-tmp/datasets/FVQA/fvqa_train.parquet")
PARQUET_SHA256 = "d23be97f4493846381f71c6953a29777fe1522aaf37942a26393605ffd78171f"
OUTPUT_DIR = Path("/root/autodl-tmp/mmsearch_step10_controlled_inputs")
SELECTION_MANIFEST = OUTPUT_DIR / "selection_manifest.json"

FIXED = {
    "A": {"data_id": "fvqa_train_0", "row": 0, "mode": "natural"},
    "B": {
        "data_id": "fvqa_train_6",
        "row": 6,
        "candidate": 1,
        "mode": "controlled_image_then_answer",
    },
    "C": {
        "data_id": "fvqa_train_9",
        "row": 9,
        "candidate": 2,
        "mode": "controlled_text_then_answer",
    },
    "D": {
        "data_id": "fvqa_train_17",
        "row": 17,
        "candidate": 5,
        "mode": "natural",
    },
    "Failure": {
        "data_id": "fvqa_train_32",
        "row": 32,
        "candidate": 8,
        "mode": "natural",
    },
}
EXPECTED_NATURAL_ROUTES = {
    ("answer",): 116,
    ("image_search", "text_search", "answer"): 396,
}
OUTPUT_BASENAMES = {
    "A": "case_a",
    "B": "case_b",
    "C": "case_c",
    "D": "case_d",
    "Failure": "failure",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path, expected_sha256: str, maximum: int) -> tuple[dict[str, Any], bytes]:
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), f"unsafe input: {path}")
    encoded = bridge.read_regular_file(path, maximum=maximum)
    require(bridge.sha256_bytes(encoded) == expected_sha256, f"SHA-256 mismatch: {path}")
    value = json.loads(encoded)
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value, encoded


def canonical_record_sha256(record: dict[str, Any]) -> str:
    return bridge.sha256_bytes(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def validate_protocol(protocol: dict[str, Any]) -> None:
    require(protocol.get("schema_version") == 1, "controlled protocol schema mismatch")
    require(
        protocol.get("status") == "registered_before_controlled_step10_execution",
        "controlled protocol status mismatch",
    )
    require(protocol.get("decision", {}).get("preserve_natural_negative_result") is True, "negative-result preservation is not pinned")
    require(protocol.get("natural_negative_evidence", {}).get("scanned_search_required_count") == 512, "natural scan count mismatch in protocol")
    cases = protocol.get("fixed_cases")
    require(isinstance(cases, dict) and set(cases) == set(FIXED), "controlled protocol case set mismatch")
    for label, fixed in FIXED.items():
        claimed = cases[label]
        require(claimed.get("data_id") == fixed["data_id"], f"protocol {label} ID mismatch")
        require(claimed.get("source_row_index") == fixed["row"], f"protocol {label} row mismatch")
        require(claimed.get("execution_mode") == fixed["mode"], f"protocol {label} mode mismatch")
        if "candidate" in fixed:
            require(claimed.get("natural_candidate_number") == fixed["candidate"], f"protocol {label} candidate mismatch")
    require(protocol.get("controlled_execution", {}).get("B", {}).get("ground_truth_may_not_be_used_by_controller") is True, "B leakage guard missing")
    require(protocol.get("controlled_execution", {}).get("C", {}).get("ground_truth_may_not_be_used_by_controller") is True, "C leakage guard missing")


def validate_natural_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(manifest.get("schema_version") == "mmsearch.step10.case-selection.v2", "natural manifest schema mismatch")
    require(manifest.get("status") == "failed_selection_not_found_within_limit", "natural manifest status mismatch")
    require(manifest.get("missing_selections") == ["case_b", "case_c"], "natural missing slots mismatch")
    require(manifest.get("credentials_recorded") is False, "natural manifest credential flag mismatch")
    require(manifest.get("artifacts") == {}, "failed natural selection unexpectedly committed artifacts")
    records = manifest.get("scan_records")
    require(isinstance(records, list) and len(records) == 512, "natural record count mismatch")
    require(manifest.get("scanned_search_required_count") == 512, "declared natural count mismatch")
    require([record.get("candidate_number") for record in records] == list(range(1, 513)), "natural candidate order mismatch")
    require(len({record.get("data_id") for record in records}) == 512, "natural data IDs are not unique")
    require(all(record.get("source_split") == "train" and record.get("category") == "search_required" for record in records), "natural pool mismatch")
    require(all(record.get("terminal_status") == "answered" for record in records), "natural terminal mismatch")
    routes = collections.Counter(tuple(record.get("action_sequence", [])) for record in records)
    require(routes == collections.Counter(EXPECTED_NATURAL_ROUTES), "natural route distribution mismatch")
    exact = collections.Counter(record.get("exact_match") for record in records)
    require(exact == collections.Counter({True: 116, False: 396}), "natural strict-EM distribution mismatch")
    checks = manifest.get("checks")
    require(isinstance(checks, dict), "natural checks missing")
    require(checks.get("v1_replay_core_match") is True, "natural v1 replay did not pass")
    require(checks.get("placeholder_bodies_not_persisted") is True, "natural body persistence check failed")
    by_id = {record["data_id"]: record for record in records}
    for label in ("B", "C", "D", "Failure"):
        fixed = FIXED[label]
        record = by_id.get(fixed["data_id"])
        require(isinstance(record, dict), f"natural record missing: {label}")
        require(record.get("candidate_number") == fixed["candidate"], f"natural candidate mismatch: {label}")
        require(record.get("source_row_index") == fixed["row"], f"natural row mismatch: {label}")
    require(by_id[FIXED["B"]["data_id"]]["action_sequence"] == ["answer"], "B natural provenance changed")
    require(by_id[FIXED["C"]["data_id"]]["action_sequence"] == ["answer"], "C natural provenance changed")
    require(by_id[FIXED["D"]["data_id"]]["action_sequence"] == ["image_search", "text_search", "answer"], "D natural provenance changed")
    failure = by_id[FIXED["Failure"]["data_id"]]
    require(failure.get("action_sequence") == ["answer"] and failure.get("exact_match") is False, "Failure natural provenance changed")
    return by_id


def load_rows() -> dict[str, tuple[int, dict[str, Any]]]:
    size, digest = bridge.hash_file(PARQUET)
    require(size > 0 and digest == PARQUET_SHA256, "FVQA parquet digest mismatch")
    wanted = {value["data_id"]: value["row"] for value in FIXED.values()}
    found: dict[str, tuple[int, dict[str, Any]]] = {}
    columns = ["prompt", "images", "reward_model", "data_source", "data_id", "category"]
    row_index = 0
    parquet = pq.ParquetFile(PARQUET)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            data_id = row.get("data_id")
            if data_id in wanted:
                require(row_index == wanted[data_id], f"parquet row mismatch: {data_id}")
                require(data_id not in found, f"duplicate parquet ID: {data_id}")
                found[data_id] = (row_index, row)
            row_index += 1
        if len(found) == len(wanted):
            break
    require(set(found) == set(wanted), "not all fixed rows were found")
    require(found[FIXED["A"]["data_id"]][1].get("category") == "search_free", "A category mismatch")
    for label in ("B", "C", "D", "Failure"):
        require(found[FIXED[label]["data_id"]][1].get("category") == "search_required", f"{label} category mismatch")
    return found


def prepare() -> dict[str, Any]:
    require(not OUTPUT_DIR.exists() and not OUTPUT_DIR.is_symlink(), "controlled input directory already exists")
    require(not SELECTION_MANIFEST.exists(), "controlled selection manifest already exists")
    protocol, protocol_bytes = load_json(PROTOCOL, PROTOCOL_SHA256, 128 * 1024)
    validate_protocol(protocol)
    natural, natural_bytes = load_json(NATURAL_MANIFEST, NATURAL_MANIFEST_SHA256, 4 * 1024 * 1024)
    natural_records = validate_natural_manifest(natural)
    rows = load_rows()

    image_payloads: dict[str, bytes] = {}
    metadata_payloads: dict[str, bytes] = {}
    output_records: dict[str, Any] = {}
    case_entries: dict[str, Any] = {}
    for label, basename in OUTPUT_BASENAMES.items():
        fixed = FIXED[label]
        row_index, row = rows[fixed["data_id"]]
        raw = bridge.row_image(row)
        normalized, _, _ = bridge.normalized_png(raw)
        image_path = OUTPUT_DIR / f"{basename}.png"
        meta_path = OUTPUT_DIR / f"{basename}.json"
        sample = bridge.suite_sample(
            row=row,
            row_index=row_index,
            image_path=image_path,
            image_bytes=normalized,
            source_sha256=bridge.sha256_bytes(raw),
        )
        image_payloads[label] = normalized
        metadata_payloads[label] = bridge.pretty_json_bytes(sample)
        output_records[label] = {
            "data_id": fixed["data_id"],
            "image": bridge.file_record(image_path, normalized),
            "metadata": bridge.file_record(meta_path, metadata_payloads[label]),
        }
        entry: dict[str, Any] = {
            "meta_file": meta_path.name,
            "selection_rule": protocol["fixed_cases"][label]["reason"],
            "execution_mode": fixed["mode"],
            "claim_scope": "natural_policy" if fixed["mode"] == "natural" else "controlled_tool_integration_only",
        }
        if label in natural_records:
            record = natural_records[label] if label in natural_records else None
        else:
            record = natural_records.get(fixed["data_id"])
        if label != "A":
            record = natural_records[fixed["data_id"]]
            entry.update({
                "candidate_number": fixed["candidate"],
                "candidate_scan_sha256": canonical_record_sha256(record),
                "natural_action_sequence": record["action_sequence"],
                "natural_exact_match": record["exact_match"],
            })
        if label == "B":
            entry["intervention"] = {
                "controller_injected_action": "image_search",
                "after_tool_allowed_action": "answer",
                "ground_truth_used": False,
            }
        elif label == "C":
            entry["intervention"] = {
                "controller_injected_action": "text_search",
                "controller_query_policy": "exact_original_question",
                "after_tool_allowed_action": "answer",
                "ground_truth_used": False,
            }
        else:
            entry["intervention"] = {"applied": False}
        case_entries[label] = entry

    selection_payload = {
        "schema_version": 1,
        "selection_rule": "User-approved controlled Step-10 completion; A/D/Failure natural, B/C fixed interventions; no correctness-based selection.",
        "dataset": {"id": bridge.DATASET_ID, "revision": bridge.DATASET_REVISION, "split": bridge.DATASET_SPLIT},
        "seed": 0,
        "cases": case_entries,
        "natural_negative_evidence": {
            "manifest": bridge.file_record(NATURAL_MANIFEST, natural_bytes),
            "status": natural["status"],
            "scanned_search_required_count": 512,
            "route_counts": {"answer": 116, "mixed": 396, "image_only": 0, "text_only": 0},
            "v1_prefix_replay_pass": True,
        },
        "controlled_protocol": bridge.file_record(PROTOCOL, protocol_bytes),
        "selected_ids": {label: value["data_id"] for label, value in FIXED.items()},
        "output_artifacts": output_records,
        "checks": {
            "controlled_protocol_pinned_before_execution": True,
            "natural_512_negative_result_preserved": True,
            "natural_manifest_sha256_verified": True,
            "natural_route_and_strict_em_counts_recomputed": True,
            "five_unique_fixed_ids": len({value["data_id"] for value in FIXED.values()}) == 5,
            "parquet_rows_and_sha256_verified": True,
            "B_C_selection_independent_of_correctness": True,
            "B_C_claim_scope_controlled_only": True,
            "credentials_read": False,
            "network_used": False,
            "model_inference_used": False,
        },
        "credentials_recorded": False,
    }
    require(all(selection_payload["checks"].values()), "controlled input check failed")
    selection_bytes = bridge.pretty_json_bytes(selection_payload)

    OUTPUT_DIR.mkdir(mode=0o700, parents=False, exist_ok=False)
    OUTPUT_DIR.chmod(0o700)
    for label, basename in OUTPUT_BASENAMES.items():
        image_path = OUTPUT_DIR / f"{basename}.png"
        meta_path = OUTPUT_DIR / f"{basename}.json"
        bridge.atomic_write(image_path, image_payloads[label])
        bridge.atomic_write(meta_path, metadata_payloads[label])
    bridge.atomic_write(SELECTION_MANIFEST, selection_bytes)
    require(bridge.read_regular_file(SELECTION_MANIFEST, maximum=4 * 1024 * 1024) == selection_bytes, "selection commit verification failed")
    return {
        "status": "passed",
        "output_dir": str(OUTPUT_DIR),
        "selection_manifest": str(SELECTION_MANIFEST),
        "selection_manifest_sha256": bridge.sha256_bytes(selection_bytes),
        "selected_ids": selection_payload["selected_ids"],
    }


def self_test() -> None:
    require(canonical_record_sha256({"b": 2, "a": 1}) == canonical_record_sha256({"a": 1, "b": 2}), "canonical digest test failed")
    require(len({value["data_id"] for value in FIXED.values()}) == 5, "fixed ID uniqueness test failed")
    require(FIXED["B"]["candidate"] == 1 and FIXED["C"]["candidate"] == 2, "controlled ordering test failed")
    print(json.dumps({"status": "passed", "pure_self_tests": 3}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
