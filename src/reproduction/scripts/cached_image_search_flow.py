#!/usr/bin/env python3
"""Run MMSearch-R1 with the official FVQA cached image-search results."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from placeholder_control_flow import (
    assistant_message,
    classify_response,
    generate_response,
    image_search_result_message,
    image_to_data_uri,
    load_prompt,
    normalize_answer,
)
from reproduction.mmsearch_tools.cached_image_search import FVQACachedImageSearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-meta", type=Path, required=True)
    parser.add_argument("--cache-pickle", type=Path, required=True)
    parser.add_argument("--thumbnail-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    sample = json.loads(args.sample_meta.read_text(encoding="utf-8"))
    question = sample["question"]
    image_path = Path(sample["image"])
    round_1_prompt = load_prompt("round_1_user_prompt_qwenvl.pkl").replace(
        "<image>", ""
    ).strip()
    after_image_search_prompt = load_prompt("after_image_search_prompt_qwenvl.pkl")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{round_1_prompt}\nQuestion: {question}\nImage: ",
                },
                {
                    "type": "image",
                    "image": image_to_data_uri(image_path),
                    "max_pixels": 672 * 672,
                },
            ],
        }
    ]

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

    first_response, first_input_tokens, first_output_tokens, first_seconds = generate_response(
        model,
        processor,
        messages,
        args.max_new_tokens,
    )
    first_action, _ = classify_response(first_response)
    print("round 1:", first_action)
    print(first_response)

    trace = {
        "data_id": sample["data_id"],
        "category": sample["category"],
        "question": question,
        "ground_truth": sample["reward_model"]["ground_truth"],
        "image": str(image_path),
        "rounds": [
            {
                "round": 1,
                "response": first_response,
                "has_reason": re.search(
                    r"<reason>.*?</reason>", first_response, flags=re.DOTALL
                )
                is not None,
                "action": first_action,
                "input_tokens": first_input_tokens,
                "output_tokens": first_output_tokens,
                "generation_seconds": round(first_seconds, 3),
            }
        ],
        "image_search_calls": 0,
        "text_search_calls": 0,
        "final_answer": None,
        "terminal_status": None,
    }

    if first_action != "image_search":
        trace["terminal_status"] = "first_round_did_not_request_image_search"
    else:
        messages.append(assistant_message(first_response))
        image_search = FVQACachedImageSearch(
            args.cache_pickle,
            args.thumbnail_cache_dir,
            top_k=5,
        )
        returned_text, returned_images, tool_stat = image_search(sample["data_id"])
        trace["image_search_calls"] = 1
        trace["rounds"][0]["tool"] = {
            "type": "fvqa_official_image_search_cache",
            "returned_text": returned_text,
            "status": tool_stat,
        }
        messages.append(
            image_search_result_message(
                returned_images,
                tool_stat["titles"],
                question,
                after_image_search_prompt,
            )
        )

        second_response, second_input_tokens, second_output_tokens, second_seconds = (
            generate_response(
                model,
                processor,
                messages,
                args.max_new_tokens,
            )
        )
        second_action, second_payload = classify_response(second_response)
        print("round 2:", second_action)
        print(second_response)
        second_round = {
            "round": 2,
            "response": second_response,
            "has_reason": re.search(
                r"<reason>.*?</reason>", second_response, flags=re.DOTALL
            )
            is not None,
            "action": second_action,
            "input_tokens": second_input_tokens,
            "output_tokens": second_output_tokens,
            "generation_seconds": round(second_seconds, 3),
        }
        if second_action == "text_search":
            second_round["query"] = second_payload
            trace["terminal_status"] = "awaiting_real_text_search"
        elif second_action == "answer":
            trace["final_answer"] = second_payload
            trace["terminal_status"] = "answered_after_image_search"
        else:
            trace["terminal_status"] = f"unexpected_second_action:{second_action}"
        trace["rounds"].append(second_round)

    trace["total_turns"] = len(trace["rounds"])
    trace["exact_match"] = (
        normalize_answer(trace["final_answer"])
        == normalize_answer(trace["ground_truth"])
        if trace["final_answer"] is not None
        else None
    )
    real_titles = []
    fetched_images = 0
    failures = []
    if trace["image_search_calls"]:
        status = trace["rounds"][0]["tool"]["status"]
        real_titles = status["titles"]
        fetched_images = status["num_images"]
        failures = status["failures"]
    trace["control_flow_pass"] = (
        first_action == "image_search"
        and trace["image_search_calls"] == 1
        and trace["total_turns"] == 2
        and fetched_images >= 1
        and len(real_titles) == fetched_images
        and all(not title.startswith("Webpage Title ") for title in real_titles)
        and trace["rounds"][1]["action"] in {"answer", "text_search"}
        and all(round_["has_reason"] for round_ in trace["rounds"])
    )

    result = {
        "mode": "fvqa_official_cached_image_search",
        "accepted_cache_degradation": {
            "requested": 5,
            "minimum_required": 1,
            "actual": fetched_images,
            "failures": failures,
        },
        "model_path": str(args.model_path),
        "attention_implementation": model.config._attn_implementation,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "trace": trace,
        "pass": trace["control_flow_pass"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
