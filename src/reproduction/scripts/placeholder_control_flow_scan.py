#!/usr/bin/env python3
"""Select a natural search-triggering FVQA sample and run placeholder flow."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from placeholder_control_flow import (
    MAX_PIXELS,
    classify_response,
    generate_response,
    load_prompt,
    run_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--samples-meta", type=Path, required=True)
    parser.add_argument("--selected-image", type=Path, required=True)
    parser.add_argument("--selected-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=3)
    return parser.parse_args()


def question_from_prompt(prompt: list[dict[str, str]]) -> str:
    for message in prompt:
        if message.get("role") == "user":
            return message["content"]
    raise ValueError("No user question in prompt")


def png_bytes(raw_image: bytes) -> tuple[bytes, int, int]:
    with Image.open(BytesIO(raw_image)) as image:
        image.load()
        image = image.convert("RGB")
        width, height = image.size
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue(), width, height


def initial_messages(round_1_prompt: str, question: str, image_png: bytes) -> list[dict[str, Any]]:
    encoded = base64.b64encode(image_png).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{round_1_prompt}\nQuestion: {question}\nImage: ",
                },
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{encoded}",
                    "max_pixels": MAX_PIXELS,
                },
            ],
        }
    ]


def search_required_rows(parquet_path: Path):
    columns = [
        "prompt",
        "images",
        "reward_model",
        "data_source",
        "data_id",
        "category",
    ]
    row_index = 0
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            if row["category"] == "search_required":
                yield row_index, row
            row_index += 1


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    fixed_samples = json.loads(args.samples_meta.read_text(encoding="utf-8"))
    round_1_prompt = load_prompt("round_1_user_prompt_qwenvl.pkl").replace(
        "<image>", ""
    ).strip()
    after_image_search_prompt = load_prompt("after_image_search_prompt_qwenvl.pkl")
    after_text_search_prompt = load_prompt("after_text_search_prompt_qwenvl.pkl")

    load_started = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        use_fast=False,
    )
    load_seconds = time.monotonic() - load_started

    scan: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_png: bytes | None = None
    for candidate_number, (source_row_index, row) in enumerate(
        search_required_rows(args.parquet),
        start=1,
    ):
        if candidate_number > args.scan_limit:
            break
        raw_image = row["images"][0]["bytes"]
        image_png, width, height = png_bytes(raw_image)
        question = question_from_prompt(row["prompt"])
        response, input_tokens, output_tokens, generation_seconds = generate_response(
            model,
            processor,
            initial_messages(round_1_prompt, question, image_png),
            args.max_new_tokens,
        )
        action, payload = classify_response(response)
        record = {
            "candidate_number": candidate_number,
            "source_row_index": source_row_index,
            "data_id": row["data_id"],
            "question": question,
            "ground_truth": row["reward_model"]["ground_truth"],
            "action": action,
            "action_payload": payload,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": round(generation_seconds, 3),
        }
        scan.append(record)
        print(f"[scan {candidate_number}/{args.scan_limit}] {row['data_id']}: {action}")
        print(response)
        if action in {"image_search", "text_search"}:
            selected_png = image_png
            selected = {
                "data_id": row["data_id"],
                "category": "search_required",
                "source_split": "train",
                "source_row_index": source_row_index,
                "data_source": row["data_source"],
                "image": str(args.selected_image),
                "image_width": width,
                "image_height": height,
                "source_image_sha256": hashlib.sha256(raw_image).hexdigest(),
                "question": question,
                "reward_model": row["reward_model"],
                "selected_first_action": action,
            }
            break

    if selected is None or selected_png is None:
        result = {
            "scan_limit": args.scan_limit,
            "scan": scan,
            "selected": None,
            "all_control_flows_pass": False,
            "terminal_status": "no_search_trigger_found",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(3)

    args.selected_image.parent.mkdir(parents=True, exist_ok=True)
    args.selected_image.write_bytes(selected_png)
    args.selected_meta.parent.mkdir(parents=True, exist_ok=True)
    args.selected_meta.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cases = [
        run_case(
            model,
            processor,
            fixed_samples["search_free"],
            round_1_prompt,
            after_image_search_prompt,
            after_text_search_prompt,
            args.max_new_tokens,
            args.max_rounds,
        ),
        run_case(
            model,
            processor,
            selected,
            round_1_prompt,
            after_image_search_prompt,
            after_text_search_prompt,
            args.max_new_tokens,
            args.max_rounds,
        ),
    ]
    result = {
        "mode": "official_placeholder_tools_with_corrected_message_history",
        "selection": "first natural search trigger among search_required rows in parquet order",
        "scan_limit": args.scan_limit,
        "scan": scan,
        "selected": selected,
        "model_path": str(args.model_path),
        "attention_implementation": model.config._attn_implementation,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "max_rounds": args.max_rounds,
        "cases": cases,
        "all_control_flows_pass": all(case["control_flow_pass"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_control_flows_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
