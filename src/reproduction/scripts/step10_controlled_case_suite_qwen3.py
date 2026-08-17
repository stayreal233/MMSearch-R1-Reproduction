#!/usr/bin/env python3
"""Run Step 10 with natural A/D/Failure and explicitly controlled B/C.

This is a thin evidence-preserving wrapper around step10_case_suite_qwen3.
It does not change the original runner.  The controller injects only the first
tool action for B/C; all real tools, post-tool model generation, health checks,
same-GPU checks, cache accounting, serialization scans, and atomic writes are
still performed by the audited base runner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import step10_case_suite_qwen3 as base


CONTROLLED_PROTOCOL = Path(
    "/root/autodl-tmp/multimodal-search-r1/"
    "reproduction/env/step10_controlled_route_protocol.json"
)
CONTROLLED_PROTOCOL_SHA256 = (
    "f18ac1cdb69e035243079c56b5abc4806609ada4d522e19ed0336d744ad5c48f"
)
NATURAL_MANIFEST = Path(
    "/root/autodl-tmp/mmsearch_step10_v2/step10_candidate_selection_v2.json"
)
NATURAL_MANIFEST_SHA256 = (
    "a512d87f3dd24fc5a2714f58589a846f00d494132e678cc2479fd5410751b17d"
)
EXPECTED_MODES = {
    "A": "natural",
    "B": "controlled_image_then_answer",
    "C": "controlled_text_then_answer",
    "D": "natural",
    "Failure": "natural",
}
EXPECTED_IDS = {
    "A": "fvqa_train_0",
    "B": "fvqa_train_6",
    "C": "fvqa_train_9",
    "D": "fvqa_train_17",
    "Failure": "fvqa_train_32",
}
CONTROLLED_AFTER_IMAGE = (
    "Controlled Step-10 Image-only integration trace. Use the image-search "
    "results to answer the original question now. Text Search is disabled for "
    "this trace. End with exactly one <answer>...</answer> element."
)
CONTROLLED_AFTER_TEXT = (
    "Controlled Step-10 Text-only integration trace. Use the text-search "
    "results to answer the original question now. Do not request another "
    "tool. End with exactly one <answer>...</answer> element."
)

_original_load_selected_samples = base.load_selected_samples
_original_execute_case = base.execute_case
_original_case_markdown = base.case_markdown
_original_atomic_write_json = base.atomic_write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    require(path.is_file() and not path.is_symlink(), f"unsafe evidence path: {path}")
    encoded = base.read_regular_file(path, max_bytes=4 * 1024 * 1024)
    require(base.sha256_bytes(encoded) == expected_sha256, f"evidence digest mismatch: {path}")
    value = json.loads(encoded)
    require(isinstance(value, dict), f"evidence root is not an object: {path}")
    return value, encoded


def controlled_load_selected_samples(
    selection_manifest: Path,
    selected_meta_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    protocol, protocol_bytes = read_json(
        CONTROLLED_PROTOCOL, CONTROLLED_PROTOCOL_SHA256
    )
    require(
        protocol.get("status") == "registered_before_controlled_step10_execution",
        "controlled protocol status mismatch",
    )
    natural, natural_bytes = read_json(NATURAL_MANIFEST, NATURAL_MANIFEST_SHA256)
    require(
        natural.get("status") == "failed_selection_not_found_within_limit"
        and natural.get("scanned_search_required_count") == 512
        and natural.get("missing_selections") == ["case_b", "case_c"],
        "natural negative evidence mismatch",
    )
    samples, selections, dataset, manifest_record = _original_load_selected_samples(
        selection_manifest, selected_meta_dir
    )
    manifest, _ = base.load_json_object(selection_manifest)
    cases = manifest.get("cases")
    require(isinstance(cases, dict), "controlled cases mapping missing")
    for label in base.CASE_ORDER:
        entry = base._manifest_case_entry(cases, label)
        mode = entry.get("execution_mode")
        scope = entry.get("claim_scope")
        intervention = entry.get("intervention")
        require(mode == EXPECTED_MODES[label], f"Case {label} execution mode mismatch")
        require(samples[label]["data_id"] == EXPECTED_IDS[label], f"Case {label} fixed ID mismatch")
        require(isinstance(intervention, dict), f"Case {label} intervention record missing")
        if mode == "natural":
            require(scope == "natural_policy" and intervention == {"applied": False}, f"Case {label} natural semantics mismatch")
        else:
            require(scope == "controlled_tool_integration_only", f"Case {label} controlled scope mismatch")
            require(intervention.get("ground_truth_used") is False, f"Case {label} leakage guard mismatch")
        selections[label].update({
            "execution_mode": mode,
            "claim_scope": scope,
            "intervention": base.sanitize_public(intervention),
            "natural_action_sequence": base.sanitize_public(
                entry.get("natural_action_sequence")
            ),
            "natural_exact_match": entry.get("natural_exact_match"),
        })
    dataset = {
        **dataset,
        "controlled_protocol": {
            "path": str(CONTROLLED_PROTOCOL),
            "bytes": len(protocol_bytes),
            "sha256": CONTROLLED_PROTOCOL_SHA256,
        },
        "natural_negative_evidence": {
            "path": str(NATURAL_MANIFEST),
            "bytes": len(natural_bytes),
            "sha256": NATURAL_MANIFEST_SHA256,
            "scanned_search_required_count": 512,
            "natural_image_only_count": 0,
            "natural_text_only_count": 0,
        },
    }
    return samples, selections, dataset, manifest_record


def controlled_execute_case(**kwargs: Any) -> dict[str, Any]:
    label = kwargs["label"]
    selection = kwargs["selection"]
    mode = selection["execution_mode"]
    require(mode == EXPECTED_MODES[label], f"Case {label} runtime mode mismatch")
    if mode == "natural":
        result = _original_execute_case(**kwargs)
        result["case_semantics"] = {
            "route_origin": "natural_model_policy",
            "controller_intervention": False,
            "claim_scope": "natural_policy",
        }
        result["trace"]["route_origin"] = "natural_model_policy"
        result["trace"]["controller_intervention"] = False
        return result

    sample = kwargs["sample"]
    original_generate = base.generate_response
    calls = 0
    if mode == "controlled_image_then_answer":
        injected_response = (
            "<reason>Controller-injected action for controlled tool-integration "
            "coverage; not a natural model policy decision.</reason>"
            "<search><img></search>"
        )
        kwargs["after_image_search_prompt"] = CONTROLLED_AFTER_IMAGE
        injected_action = "image_search"
        query_policy = None
    else:
        query = sample["question"]
        require(
            isinstance(query, str)
            and query.strip()
            and "</text_search>" not in query.lower(),
            "controlled C original-question query is unsafe",
        )
        injected_response = (
            "<reason>Controller-injected action for controlled tool-integration "
            "coverage; not a natural model policy decision.</reason>"
            f"<text_search>{query}</text_search>"
        )
        kwargs["after_text_search_prompt"] = CONTROLLED_AFTER_TEXT
        injected_action = "text_search"
        query_policy = "exact_original_question"

    def generate_with_controlled_first_action(*args: Any, **inner_kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 1:
            return injected_response, 0, 0, 0.0
        return original_generate(*args, **inner_kwargs)

    base.generate_response = generate_with_controlled_first_action
    try:
        result = _original_execute_case(**kwargs)
    finally:
        base.generate_response = original_generate

    first_round = result["trace"]["rounds"][0]
    require(first_round.get("action") == injected_action, f"Case {label} injected action was not preserved")
    require(
        first_round.get("input_tokens") == 0
        and first_round.get("output_tokens") == 0
        and first_round.get("generation_seconds") == 0.0,
        f"Case {label} injected action token provenance mismatch",
    )
    first_round["origin"] = "controller_intervention"
    first_round["model_generated"] = False
    result["trace"]["route_origin"] = "controlled_tool_integration"
    result["trace"]["controller_intervention"] = True
    result["trace"]["controller_intervention_detail"] = {
        "injected_round": 1,
        "injected_action": injected_action,
        "query_policy": query_policy,
        "ground_truth_used": False,
        "post_tool_model_generation": True,
    }
    result["case_semantics"] = {
        "route_origin": "controlled_tool_integration",
        "controller_intervention": True,
        "claim_scope": "controlled_tool_integration_only",
        "must_not_be_used_for_search_ratio": True,
        "natural_scan_action_sequence": selection.get("natural_action_sequence"),
    }
    return result


def controlled_case_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# MMSearch-R1 Step 10 cases",
        "",
        "A, D, and Failure are natural model-policy traces. B and C are explicitly controlled tool-integration traces after the immutable 512-candidate natural scan produced zero B/C routes. B/C must not be used for Search Ratio or described as natural policy selections.",
        "",
        "| Case | semantics | data_id | actions | path pass | EM | terminal |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for result in results:
        trace = result["trace"]
        actions = " → ".join(trace["action_sequence"]) or "(none)"
        semantics = result["case_semantics"]["route_origin"]
        path_pass = trace.get("path_pass")
        path_display = "n/a" if path_pass is None else str(path_pass).lower()
        lines.append(
            f"| {result['case_label']} ({result['case_type']}) | {semantics} | "
            f"{trace['data_id']} | {actions} | {path_display} | "
            f"{str(trace['exact_match']).lower()} | {trace['terminal_status']} |"
        )
    failure = results[-1]["trace"].get("failure_analysis", {})
    lines.extend([
        "",
        "## Natural-policy negative result",
        "",
        "- Fixed FVQA train/search_required candidates scanned: `512`",
        "- Natural routes: `116 answer`, `396 image_search → text_search → answer`",
        "- Natural Image-only routes: `0`",
        "- Natural Text-only routes: `0`",
        "- Evidence: `/root/autodl-tmp/mmsearch_step10_v2/step10_candidate_selection_v2.json`",
        "",
        "## Failure case",
        "",
        f"- data_id: `{results[-1]['trace']['data_id']}`",
        f"- classified layer: `{failure.get('layer')}`",
        f"- rationale: {failure.get('rationale')}",
        f"- evidence paths: `{', '.join(failure.get('evidence_paths', []))}`",
        f"- infrastructure failure: `{str(failure.get('infrastructure_failure')).lower()}`",
        f"- terminal status: `{results[-1]['trace']['terminal_status']}`",
        "",
    ])
    return "\n".join(lines)


def controlled_atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.name == "step10_completion_manifest.json":
        payload = dict(payload)
        payload["completion_scope"] = (
            "engineering_path_coverage_with_controlled_B_C; natural policy B/C remain unobserved"
        )
        payload["natural_policy_result"] = {
            "status": "incomplete_four_route_coverage",
            "scanned_search_required_count": 512,
            "natural_route_counts": {
                "answer": 116,
                "mixed": 396,
                "image_only": 0,
                "text_only": 0,
            },
            "manifest": {
                "path": str(NATURAL_MANIFEST),
                "sha256": NATURAL_MANIFEST_SHA256,
            },
        }
        payload["case_semantics"] = {
            "A": "natural_model_policy",
            "B": "controlled_tool_integration_only",
            "C": "controlled_tool_integration_only",
            "D": "natural_model_policy",
            "Failure": "natural_model_policy",
        }
        producer_checks = dict(payload.get("producer_checks", {}))
        producer_checks.update({
            "natural_negative_result_preserved": True,
            "controlled_B_C_not_claimed_as_natural": True,
            "controlled_B_C_excluded_from_search_ratio": True,
        })
        payload["producer_checks"] = producer_checks
    _original_atomic_write_json(path, payload)


def self_test() -> None:
    require(set(EXPECTED_MODES) == set(base.CASE_ORDER), "mode labels mismatch")
    require(len(set(EXPECTED_IDS.values())) == 5, "controlled IDs are not unique")
    require(EXPECTED_MODES["B"].startswith("controlled_"), "B mode test failed")
    require(EXPECTED_MODES["C"].startswith("controlled_"), "C mode test failed")
    print(json.dumps({"status": "passed", "pure_self_tests": 4}, sort_keys=True))


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    base.load_selected_samples = controlled_load_selected_samples
    base.execute_case = controlled_execute_case
    base.case_markdown = controlled_case_markdown
    base.atomic_write_json = controlled_atomic_write_json
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
