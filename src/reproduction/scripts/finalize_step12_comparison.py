#!/usr/bin/env python3
"""Build the paired Base/MMSearch comparison and final Step 12 report."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "reproduction" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import step10_case_suite_qwen3 as evidence  # noqa: E402


PROTOCOL = REPO_ROOT / "reproduction/env/step12_base_comparison_protocol.json"
PROTOCOL_SHA256 = "3a80ee1fe4685cde68335a1ad336a3cf6f8f970a71f9664f6f66e22ee3d651f5"
BASE_ROOT = Path("/root/autodl-tmp/outputs/step12_base_direct_v1")
BASE_COMPLETION = BASE_ROOT / "step12_base_completion_manifest.json"
BASE_PREDICTIONS = BASE_ROOT / "predictions.jsonl"
BASE_METRICS = BASE_ROOT / "metrics.json"
MM_ROOT = Path("/root/autodl-tmp/outputs/step11_eval_v2")
MM_COMPLETION = MM_ROOT / "step11_completion_manifest.json"
MM_PREDICTIONS = MM_ROOT / "predictions.jsonl"
MM_METRICS = MM_ROOT / "metrics.json"
MM_COMPLETION_SHA256 = "f2747c945a022d578e3d053e7112a05937b0267894dc03b6f74adeabefe2ad87"
MM_PREDICTIONS_SHA256 = "beda0f6a02f750b89a5a53cb9f39dcaac56dbe047e2fb1f4ecc01290d6ac48ff"
MM_METRICS_SHA256 = "ec7c289417defc57ddd5a731e6bd1a2cb0a498013d281df863ad0aaccc5f0445"
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/step12_comparison_v1")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, maximum: int = 32 * 1024 * 1024) -> dict[str, Any]:
    return evidence.load_json_object(path, max_bytes=maximum)[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    encoded = evidence.read_regular_file(path, max_bytes=128 * 1024 * 1024)
    lines = encoded.decode("utf-8").splitlines()
    values = [json.loads(line) for line in lines if line.strip()]
    require(all(isinstance(item, dict) for item in values), f"JSONL object invalid: {path}")
    return values


def verify_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    record = evidence.file_record(path)
    require(record["sha256"] == expected, f"{label} SHA mismatch")
    return record


def run_command(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=120)
    return result.stdout


def outcome(base_correct: bool, mm_correct: bool) -> str:
    if base_correct and mm_correct:
        return "both_correct"
    if base_correct:
        return "base_only_correct"
    if mm_correct:
        return "mmsearch_only_correct"
    return "both_wrong"


def paired_metrics(base_values: list[dict[str, Any]], mm_values: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(len(base_values) == len(mm_values) == 50, "paired prediction count must be 50")
    paired = []
    for expected, (base_item, mm_item) in enumerate(zip(base_values, mm_values, strict=True), start=1):
        trace = mm_item.get("trace", {})
        require(base_item.get("eval_index") == mm_item.get("eval_index") == expected, f"paired eval index mismatch at {expected}")
        require(base_item.get("data_id") == trace.get("data_id"), f"paired data ID mismatch at {expected}")
        require(base_item.get("category") == trace.get("category"), f"paired category mismatch at {expected}")
        require(base_item.get("result", {}).get("ground_truth") == trace.get("ground_truth"), f"paired ground truth mismatch at {expected}")
        base_correct = base_item["result"]["strict_exact_match"] is True
        mm_correct = trace["exact_match"] is True
        paired.append({
            "eval_index": expected,
            "data_id": base_item["data_id"],
            "category": base_item["category"],
            "ground_truth": base_item["result"]["ground_truth"],
            "base_answer": base_item["result"]["answer"],
            "base_correct": base_correct,
            "base_parser_mode": base_item["result"]["parser_mode"],
            "base_generation_seconds": base_item["runtime"]["generation_seconds"],
            "base_case_seconds": base_item["runtime"]["case_seconds"],
            "base_search_calls": 0,
            "mmsearch_answer": trace.get("final_answer"),
            "mmsearch_correct": mm_correct,
            "mmsearch_generation_seconds": trace.get("mmsearch_generation_seconds", 0.0),
            "mmsearch_case_seconds": trace.get("case_seconds", 0.0),
            "mmsearch_search_calls": trace.get("image_search_calls", 0) + trace.get("text_search_calls", 0),
            "mmsearch_tool_infrastructure_success": trace.get("tool_infrastructure_success"),
            "outcome": outcome(base_correct, mm_correct),
        })
    counts = collections.Counter(item["outcome"] for item in paired)
    category_metrics = {}
    for category in ("search_free", "search_required"):
        selected = [item for item in paired if item["category"] == category]
        base_correct = sum(item["base_correct"] for item in selected)
        mm_correct = sum(item["mmsearch_correct"] for item in selected)
        category_metrics[category] = {
            "count": len(selected),
            "base_correct": base_correct,
            "base_accuracy_percent": round(base_correct / len(selected) * 100, 4),
            "mmsearch_correct": mm_correct,
            "mmsearch_accuracy_percent": round(mm_correct / len(selected) * 100, 4),
            "mmsearch_minus_base_accuracy_points": round((mm_correct - base_correct) / len(selected) * 100, 4),
            "paired_outcomes": dict(sorted(collections.Counter(item["outcome"] for item in selected).items())),
        }
    base_correct = sum(item["base_correct"] for item in paired)
    mm_correct = sum(item["mmsearch_correct"] for item in paired)
    base_generation = sum(float(item["base_generation_seconds"]) for item in paired)
    mm_generation = sum(float(item["mmsearch_generation_seconds"]) for item in paired)
    base_case = sum(float(item["base_case_seconds"]) for item in paired)
    mm_case = sum(float(item["mmsearch_case_seconds"]) for item in paired)
    metrics = {
        "schema_version": "mmsearch.step12.paired-comparison.v1",
        "evaluated": 50,
        "strict_exact_match_definition": "prediction.strip().lower() == ground_truth.strip().lower()",
        "base": {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "mode": "direct_answer_no_tools",
            "correct": base_correct,
            "accuracy_percent": round(base_correct / 50 * 100, 4),
            "search_calls": 0,
            "search_ratio_percent": 0.0,
            "average_generation_seconds": round(base_generation / 50, 6),
            "average_case_seconds": round(base_case / 50, 6),
        },
        "mmsearch": {
            "model": "lmms-lab/MMSearch-R1-7B",
            "revision": "3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46",
            "mode": "natural_on_demand_search",
            "correct": mm_correct,
            "accuracy_percent": round(mm_correct / 50 * 100, 4),
            "search_calls": sum(item["mmsearch_search_calls"] for item in paired),
            "search_ratio_percent": round(sum(item["mmsearch_search_calls"] for item in paired) / 100 * 100, 4),
            "average_generation_seconds": round(mm_generation / 50, 6),
            "average_case_seconds": round(mm_case / 50, 6),
            "sample_tool_failure_count": sum(item["mmsearch_tool_infrastructure_success"] is not True for item in paired),
        },
        "mmsearch_minus_base_accuracy_points": round((mm_correct - base_correct) / 50 * 100, 4),
        "paired_outcomes": {
            "both_correct": counts.get("both_correct", 0),
            "base_only_correct": counts.get("base_only_correct", 0),
            "mmsearch_only_correct": counts.get("mmsearch_only_correct", 0),
            "both_wrong": counts.get("both_wrong", 0),
        },
        "categories": category_metrics,
        "fairness": {
            "same_fixed_examples": True,
            "same_images_questions_ground_truth_order": True,
            "same_seed_do_sample_max_new_tokens_strict_em": True,
            "base_has_tools": False,
            "mmsearch_natural_tools": True,
            "different_role_specific_prompts": True,
            "controlled_step10_cases_excluded": True,
        },
    }
    return metrics, paired


def build_examples(paired: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for label in ("both_correct", "base_only_correct", "mmsearch_only_correct", "both_wrong"):
        groups[label] = [item for item in paired if item["outcome"] == label][:3]
    return {
        "schema_version": "mmsearch.step12.paired-examples.v1",
        "selection": "first three by fixed eval_index within each paired outcome; no correctness-based replacement beyond the declared outcome grouping",
        "groups": groups,
        "credentials_recorded": False,
    }


def report_text(metrics: dict[str, Any]) -> str:
    b = metrics["base"]
    m = metrics["mmsearch"]
    o = metrics["paired_outcomes"]
    sf = metrics["categories"]["search_free"]
    sr = metrics["categories"]["search_required"]
    return f"""# MMSearch-R1 Step 12：Base 公平对比与最终报告

生成时间：{utc_now()}

## 对比范围

- 同一固定 50 条 FVQA train 子集，Search-Free/Search-Required 各 25 条；
- Base：Qwen2.5-VL-7B-Instruct Direct Answer，零工具；
- MMSearch：MMSearch-R1-7B 自然按需搜索，复用大步 11 不可变结果；
- strict EM：`prediction.strip().lower() == ground_truth.strip().lower()`；
- 大步 10 受控 B/C 不进入本对比。

## 核心结果

| 模型 | Correct | Accuracy | Search Calls | Search Ratio |
|---|---:|---:|---:|---:|
| Base Direct Answer | {b['correct']}/50 | {b['accuracy_percent']}% | 0 | 0.0% |
| MMSearch On-demand Search | {m['correct']}/50 | {m['accuracy_percent']}% | {m['search_calls']} | {m['search_ratio_percent']}% |

MMSearch 相对 Base 的准确率差：{metrics['mmsearch_minus_base_accuracy_points']} 个百分点。

| 类别 | Base Accuracy | MMSearch Accuracy | MMSearch-Base |
|---|---:|---:|---:|
| Search-Free | {sf['base_accuracy_percent']}% | {sf['mmsearch_accuracy_percent']}% | {sf['mmsearch_minus_base_accuracy_points']} pp |
| Search-Required | {sr['base_accuracy_percent']}% | {sr['mmsearch_accuracy_percent']}% | {sr['mmsearch_minus_base_accuracy_points']} pp |

## 配对结果

- Both correct：{o['both_correct']}；
- Base only correct：{o['base_only_correct']}；
- MMSearch only correct：{o['mmsearch_only_correct']}；
- Both wrong：{o['both_wrong']}。

## 延迟与基础设施口径

- Base 平均模型生成时间：{b['average_generation_seconds']} 秒；
- MMSearch 平均模型生成时间：{m['average_generation_seconds']} 秒；
- Base 平均端到端 case 时间：{b['average_case_seconds']} 秒；
- MMSearch 平均端到端 case 时间：{m['average_case_seconds']} 秒；
- MMSearch 样本级工具失败：{m['sample_tool_failure_count']} 条，均已按大步 11 主口径计为错误；
- Base 不调用外部工具，因此 Search Ratio 固定为 0。

## 公平性与限制

1. 两模型使用相同图片、问题、Ground Truth、顺序、seed、greedy、max_new_tokens 和 strict EM；
2. Prompt 因角色不同而不同：Base 接受 Direct Answer 指令，MMSearch 接受其训练时工具控制指令；
3. 工具权限差异是本实验的设计目标，不是隐藏偏差；
4. 本子集来自 FVQA train 的确定性平衡 50 条，不是论文完整 benchmark；
5. MMSearch 结果受实时缩略图、Serper、Jina 与摘要服务状态影响，Base 不受这些外部工具影响；
6. 本报告是开源权重推理/系统复现，不是 GRPO/veRL 训练复现；
7. 未执行 LLM-as-Judge，主指标保持 strict EM。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(outcome(True, True) == "both_correct", "outcome self-test failed")
        require(outcome(True, False) == "base_only_correct", "outcome self-test failed")
        require(outcome(False, True) == "mmsearch_only_correct", "outcome self-test failed")
        require(outcome(False, False) == "both_wrong", "outcome self-test failed")
        print(json.dumps({"status": "passed", "pure_self_tests": 4}))
        return

    verify_hash(PROTOCOL, PROTOCOL_SHA256, "Step 12 protocol")
    verify_hash(MM_COMPLETION, MM_COMPLETION_SHA256, "MMSearch completion")
    verify_hash(MM_PREDICTIONS, MM_PREDICTIONS_SHA256, "MMSearch predictions")
    verify_hash(MM_METRICS, MM_METRICS_SHA256, "MMSearch metrics")
    require(not OUTPUT_ROOT.exists(), "comparison output root already exists")
    OUTPUT_ROOT.mkdir(mode=0o700, parents=False, exist_ok=False)
    base_completion = read_json(BASE_COMPLETION)
    require(base_completion.get("status") == "passed" and base_completion.get("completed_examples") == 50, "Base completion not passed")
    base_values = read_jsonl(BASE_PREDICTIONS)
    mm_values = read_jsonl(MM_PREDICTIONS)
    metrics, paired = paired_metrics(base_values, mm_values)
    base_metrics = read_json(BASE_METRICS)
    mm_metrics = read_json(MM_METRICS)
    require(metrics["base"]["correct"] == base_metrics["correct"], "Base metric recomputation mismatch")
    require(metrics["base"]["accuracy_percent"] == base_metrics["accuracy_percent"], "Base accuracy recomputation mismatch")
    require(metrics["mmsearch"]["correct"] == mm_metrics["correct"], "MMSearch metric recomputation mismatch")
    require(metrics["mmsearch"]["accuracy_percent"] == mm_metrics["accuracy_percent"], "MMSearch accuracy recomputation mismatch")
    require(metrics["mmsearch"]["search_calls"] == mm_metrics["total_search_calls"], "MMSearch search-call mismatch")
    require(metrics["mmsearch"]["search_ratio_percent"] == mm_metrics["search_ratio_percent"], "MMSearch Search Ratio mismatch")

    comparison_path = OUTPUT_ROOT / "comparison_metrics.json"
    paired_path = OUTPUT_ROOT / "paired_outcomes.jsonl"
    examples_path = OUTPUT_ROOT / "success_failure_examples.json"
    report_path = OUTPUT_ROOT / "final_report.md"
    evidence.atomic_write_json(comparison_path, metrics)
    evidence.atomic_write_text(paired_path, "\n".join(json.dumps(evidence.sanitize_public(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in paired) + "\n")
    evidence.atomic_write_json(examples_path, build_examples(paired))
    evidence.atomic_write_text(report_path, report_text(metrics))

    pip_freeze = run_command([sys.executable, "-m", "pip", "freeze"])
    pip_check = run_command([sys.executable, "-m", "pip", "check"])
    gpu_info = run_command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"])
    compute_info = run_command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    main_commit = run_command(["git", "rev-parse", "HEAD"], REPO_ROOT).strip()
    verl_commit = run_command(["git", "-C", str(REPO_ROOT / "verl"), "rev-parse", "HEAD"]).strip()
    git_status = run_command(["git", "status", "--short", "--", "reproduction"], REPO_ROOT)
    pip_path = OUTPUT_ROOT / "pip_freeze.txt"
    gpu_path = OUTPUT_ROOT / "gpu_info.txt"
    git_path = OUTPUT_ROOT / "git_commit.txt"
    revisions_path = OUTPUT_ROOT / "model_and_dataset_revisions.json"
    evidence.atomic_write_text(pip_path, pip_freeze)
    evidence.atomic_write_text(gpu_path, f"GPU:\n{gpu_info}Compute processes:\n{compute_info}")
    evidence.atomic_write_text(git_path, f"main={main_commit}\nverl={verl_commit}\nstatus_reproduction:\n{git_status}")
    evidence.atomic_write_json(revisions_path, {
        "schema_version": 1,
        "dataset": {"repo_id": "lmms-lab/FVQA", "revision": "bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5", "split": "train"},
        "base": {"repo_id": "Qwen/Qwen2.5-VL-7B-Instruct", "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5"},
        "mmsearch": {"repo_id": "lmms-lab/MMSearch-R1-7B", "revision": "3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46"},
        "summarizer": {"repo_id": "Qwen/Qwen3-32B-FP8", "revision": "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"},
        "main_git_commit": main_commit,
        "verl_git_commit": verl_commit,
        "pip_check": pip_check.strip(),
        "credentials_recorded": False,
    })

    json_paths = [comparison_path, examples_path, revisions_path]
    text_paths = [paired_path, report_path, pip_path, gpu_path, git_path]
    scan = evidence.scan_serialized_evidence(json_paths, text_paths)
    require(scan.get("pass") is True, "Step 12 comparison evidence scan failed")
    artifacts = [evidence.file_record(path) for path in [*json_paths, *text_paths]]
    completion = {
        "schema_version": "mmsearch.step12.completion.v1",
        "status": "passed",
        "step": 12,
        "completed_at_utc": utc_now(),
        "scope": "Base Direct Answer versus MMSearch natural on-demand search on the same fixed 50 examples",
        "protocol": evidence.file_record(PROTOCOL),
        "base_completion": evidence.file_record(BASE_COMPLETION),
        "mmsearch_completion": evidence.file_record(MM_COMPLETION),
        "base_predictions": evidence.file_record(BASE_PREDICTIONS),
        "mmsearch_predictions": evidence.file_record(MM_PREDICTIONS),
        "comparison_metrics": metrics,
        "artifacts": artifacts,
        "serialized_evidence_scan": scan,
        "training_reproduction": False,
        "credentials_recorded": False,
    }
    completion_path = OUTPUT_ROOT / "step12_completion_manifest.json"
    evidence.atomic_write_json(completion_path, completion)
    print(json.dumps({
        "status": "passed",
        "base_accuracy_percent": metrics["base"]["accuracy_percent"],
        "mmsearch_accuracy_percent": metrics["mmsearch"]["accuracy_percent"],
        "mmsearch_minus_base_accuracy_points": metrics["mmsearch_minus_base_accuracy_points"],
        "output": str(completion_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
