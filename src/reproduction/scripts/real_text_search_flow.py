#!/usr/bin/env python3
"""Run the full cached-image / Serper / Jina MMSearch-R1 control flow."""

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
    text_search_result_message,
)
from reproduction.mmsearch_tools.cached_image_search import FVQACachedImageSearch
from reproduction.mmsearch_tools.real_text_search import SerperJinaTextSearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-meta", type=Path, required=True)
    parser.add_argument("--cache-pickle", type=Path, required=True)
    parser.add_argument("--thumbnail-cache-dir", type=Path, required=True)
    parser.add_argument("--serper-cache-dir", type=Path, required=True)
    parser.add_argument("--jina-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars-per-page", type=int, default=12_000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def has_reason(response: str) -> bool:
    return re.search(r"<reason>.*?</reason>", response, flags=re.DOTALL) is not None


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    sample = json.loads(args.sample_meta.read_text(encoding="utf-8"))
    question = sample["question"]
    image_path = Path(sample["image"])
    ground_truth = sample["reward_model"]["ground_truth"]
    round_1_prompt = load_prompt("round_1_user_prompt_qwenvl.pkl").replace(
        "<image>", ""
    ).strip()
    after_image_search_prompt = load_prompt("after_image_search_prompt_qwenvl.pkl")
    after_text_search_prompt = load_prompt("after_text_search_prompt_qwenvl.pkl")
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

    trace = {
        "data_id": sample["data_id"],
        "category": sample["category"],
        "question": question,
        "ground_truth": ground_truth,
        "image": str(image_path),
        "rounds": [],
        "image_search_calls": 0,
        "text_search_calls": 0,
        "final_answer": None,
        "terminal_status": None,
    }

    first_response, input_tokens, output_tokens, seconds = generate_response(
        model, processor, messages, args.max_new_tokens
    )
    first_action, _ = classify_response(first_response)
    first_round = {
        "round": 1,
        "response": first_response,
        "has_reason": has_reason(first_response),
        "action": first_action,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_seconds": round(seconds, 3),
    }
    trace["rounds"].append(first_round)
    print(f"round 1: {first_action}\n{first_response}")

    if first_action != "image_search":
        trace["terminal_status"] = "first_round_did_not_request_image_search"
    else:
        messages.append(assistant_message(first_response))
        image_search = FVQACachedImageSearch(
            args.cache_pickle,
            args.thumbnail_cache_dir,
            top_k=args.top_k,
        )
        image_text, images, image_stat = image_search(sample["data_id"])
        trace["image_search_calls"] = 1
        first_round["tool"] = {
            "type": "fvqa_official_image_search_cache",
            "returned_text": image_text,
            "status": image_stat,
        }
        messages.append(
            image_search_result_message(
                images,
                image_stat["titles"],
                question,
                after_image_search_prompt,
            )
        )

        second_response, input_tokens, output_tokens, seconds = generate_response(
            model, processor, messages, args.max_new_tokens
        )
        second_action, second_payload = classify_response(second_response)
        second_round = {
            "round": 2,
            "response": second_response,
            "has_reason": has_reason(second_response),
            "action": second_action,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": round(seconds, 3),
        }
        trace["rounds"].append(second_round)
        print(f"round 2: {second_action}\n{second_response}")

        if second_action != "text_search":
            trace["terminal_status"] = f"second_round_not_text_search:{second_action}"
        else:
            query = second_payload or ""
            second_round["query"] = query
            text_search = SerperJinaTextSearch(
                args.serper_cache_dir,
                args.jina_cache_dir,
                top_k=args.top_k,
                max_chars_per_page=args.max_chars_per_page,
            )
            text, text_stat = text_search(query)
            trace["text_search_calls"] = 1
            second_round["tool"] = {
                "type": "serper_dev_plus_jina_reader",
                "status": text_stat,
                "returned_text": text,
            }
            messages.append(assistant_message(second_response))
            messages.append(
                text_search_result_message(
                    text,
                    question,
                    after_text_search_prompt,
                )
            )

            third_response, input_tokens, output_tokens, seconds = generate_response(
                model, processor, messages, args.max_new_tokens
            )
            third_action, third_payload = classify_response(third_response)
            third_round = {
                "round": 3,
                "response": third_response,
                "has_reason": has_reason(third_response),
                "action": third_action,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "generation_seconds": round(seconds, 3),
            }
            trace["rounds"].append(third_round)
            print(f"round 3: {third_action}\n{third_response}")
            if third_action == "answer":
                trace["final_answer"] = third_payload
                trace["terminal_status"] = "answered"
            else:
                trace["terminal_status"] = f"third_round_not_answer:{third_action}"

    trace["total_turns"] = len(trace["rounds"])
    trace["action_sequence"] = [item["action"] for item in trace["rounds"]]
    trace["exact_match"] = (
        normalize_answer(trace["final_answer"]) == normalize_answer(ground_truth)
        if trace["final_answer"] is not None
        else False
    )
    text_stat = (
        trace["rounds"][1].get("tool", {}).get("status", {})
        if len(trace["rounds"]) >= 2
        else {}
    )
    control_flow_pass = (
        trace["action_sequence"] == ["image_search", "text_search", "answer"]
        and trace["image_search_calls"] == 1
        and trace["text_search_calls"] == 1
        and trace["terminal_status"] == "answered"
        and trace["exact_match"]
        and all(item["has_reason"] for item in trace["rounds"])
        and text_stat.get("search", {}).get("num_results") == args.top_k
        and text_stat.get("reader", {}).get("num_documents", 0) >= 1
    )
    trace["control_flow_pass"] = control_flow_pass

    result = {
        "mode": "cached_image_search_plus_serper_dev_plus_raw_jina",
        "model_path": str(args.model_path),
        "attention_implementation": model.config._attn_implementation,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "top_k": args.top_k,
        "max_chars_per_page": args.max_chars_per_page,
        "trace": trace,
        "pass": control_flow_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "action_sequence": trace["action_sequence"],
                "final_answer": trace["final_answer"],
                "ground_truth": ground_truth,
                "exact_match": trace["exact_match"],
                "peak_gpu_memory_mib": result["peak_gpu_memory_mib"],
                "output": str(args.output),
                "pass": result["pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
