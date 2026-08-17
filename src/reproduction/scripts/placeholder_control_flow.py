#!/usr/bin/env python3
"""Validate MMSearch-R1 multi-turn control flow with the official placeholders."""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from mmsearch_r1.utils.tools.image_search import call_image_search
from mmsearch_r1.utils.tools.text_search import call_text_search


MAX_PIXELS = 672 * 672
IMAGE_SEARCH_LIMIT = 1
TEXT_SEARCH_LIMIT = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--samples-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=3)
    return parser.parse_args()


def load_prompt(name: str) -> str:
    path = Path("mmsearch_r1/prompts") / name
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, str):
        raise TypeError(f"Expected string prompt in {path}, got {type(value).__name__}")
    return value


def image_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pil_to_data_uri(image: Any) -> str:
    buffer = BytesIO()
    image_format = image.format or "PNG"
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "jpeg" if image_format.upper() in {"JPG", "JPEG"} else image_format.lower()
    return f"data:image/{mime};base64,{encoded}"


def normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def classify_response(response: str) -> tuple[str, str | None]:
    stripped = response.strip()
    if re.search(r"<search><img></search>\s*$", stripped):
        return "image_search", None
    text_match = re.search(
        r"<text_search>(.*?)</text_search>\s*$",
        stripped,
        flags=re.DOTALL,
    )
    if text_match:
        return "text_search", text_match.group(1).strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", stripped, flags=re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()
    if "Unable to answer due to lack of relevant information" in stripped:
        return "warning", None
    return "invalid", None


def generate_response(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    started = time.monotonic()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    generated_only = generated_ids[:, inputs.input_ids.shape[-1] :]
    response = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response, int(inputs.input_ids.shape[-1]), int(generated_only.shape[-1]), elapsed


def assistant_message(response: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": response}],
    }


def image_search_result_message(
    images: list[Any],
    titles: list[str],
    question: str,
    after_image_search_prompt: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Search Results: <information> [Image search results] The result of the "
                "image search consists of web page information related to the image from "
                "the user's original question. Each result includes the main image from "
                "the web page and its title, ranked in descending order of search relevance, "
                "as demonstrated below:"
            ),
        }
    ]
    for image, title in zip(images, titles):
        content.extend(
            [
                {
                    "type": "image",
                    "image": pil_to_data_uri(image),
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": f"Title: {title}"},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"</information> Original user's question: {question}\n"
                f"{after_image_search_prompt}"
            ),
        }
    )
    return {"role": "user", "content": content}


def text_search_result_message(
    result: str,
    question: str,
    after_text_search_prompt: str,
) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Search Results: <information>{result}</information> "
                    f"Original question: {question}\n{after_text_search_prompt}"
                ),
            }
        ],
    }


def run_case(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    sample: dict[str, Any],
    round_1_prompt: str,
    after_image_search_prompt: str,
    after_text_search_prompt: str,
    max_new_tokens: int,
    max_rounds: int,
) -> dict[str, Any]:
    image_path = Path(sample["image"])
    question = sample["question"]
    messages: list[dict[str, Any]] = [
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
                    "max_pixels": MAX_PIXELS,
                },
            ],
        }
    ]
    trace: dict[str, Any] = {
        "data_id": sample["data_id"],
        "category": sample["category"],
        "question": question,
        "ground_truth": sample["reward_model"]["ground_truth"],
        "image": str(image_path),
        "rounds": [],
        "image_search_calls": 0,
        "text_search_calls": 0,
        "final_answer": None,
        "terminal_status": None,
    }

    for round_number in range(1, max_rounds + 1):
        response, input_tokens, output_tokens, generation_seconds = generate_response(
            model,
            processor,
            messages,
            max_new_tokens,
        )
        action, payload = classify_response(response)
        round_trace: dict[str, Any] = {
            "round": round_number,
            "response": response,
            "has_reason": re.search(r"<reason>.*?</reason>", response, flags=re.DOTALL)
            is not None,
            "action": action,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": round(generation_seconds, 3),
        }
        trace["rounds"].append(round_trace)
        print(f"[{sample['category']}] round {round_number}: {action}")
        print(response)

        if action == "answer":
            trace["final_answer"] = payload
            trace["terminal_status"] = "answered"
            break
        if action == "warning":
            trace["terminal_status"] = "warning"
            break
        if action == "invalid":
            trace["terminal_status"] = "invalid_response"
            break

        # Preserve the model's tool action before appending the tool's user message.
        messages.append(assistant_message(response))

        if action == "image_search":
            if trace["image_search_calls"] >= IMAGE_SEARCH_LIMIT:
                trace["terminal_status"] = "image_search_limit"
                break
            returned_text, returned_images, tool_stat = call_image_search(
                image_url=str(image_path)
            )
            trace["image_search_calls"] += 1
            titles = tool_stat.get(
                "titles",
                [f"Webpage Title {index + 1}" for index in range(len(returned_images))],
            )
            round_trace["tool"] = {
                "type": "official_placeholder_image_search",
                "status": tool_stat,
                "returned_text": returned_text,
                "titles": titles,
                "returned_images": len(returned_images),
            }
            messages.append(
                image_search_result_message(
                    returned_images,
                    titles,
                    question,
                    after_image_search_prompt,
                )
            )
            continue

        if action == "text_search":
            if trace["text_search_calls"] >= TEXT_SEARCH_LIMIT:
                trace["terminal_status"] = "text_search_limit"
                break
            query = payload or ""
            returned_text, tool_stat = call_text_search(query)
            trace["text_search_calls"] += 1
            round_trace["query"] = query
            round_trace["tool"] = {
                "type": "official_placeholder_text_search",
                "status": tool_stat,
                "returned_text": returned_text,
            }
            messages.append(
                text_search_result_message(
                    returned_text,
                    question,
                    after_text_search_prompt,
                )
            )
            continue
    else:
        trace["terminal_status"] = "max_rounds"

    trace["total_turns"] = len(trace["rounds"])
    trace["exact_match"] = (
        normalize_answer(trace["final_answer"])
        == normalize_answer(trace["ground_truth"])
        if trace["final_answer"] is not None
        else False
    )
    if trace["category"] == "search_free":
        trace["control_flow_pass"] = (
            trace["terminal_status"] == "answered"
            and trace["total_turns"] == 1
            and trace["image_search_calls"] == 0
            and trace["text_search_calls"] == 0
        )
    else:
        trace["control_flow_pass"] = (
            trace["total_turns"] >= 2
            and trace["image_search_calls"] + trace["text_search_calls"] >= 1
            and all(round_["has_reason"] for round_ in trace["rounds"])
        )
    return trace


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    samples = json.loads(args.samples_meta.read_text(encoding="utf-8"))
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

    traces = []
    for category in ("search_free", "search_required"):
        traces.append(
            run_case(
                model,
                processor,
                samples[category],
                round_1_prompt,
                after_image_search_prompt,
                after_text_search_prompt,
                args.max_new_tokens,
                args.max_rounds,
            )
        )

    result = {
        "mode": "official_placeholder_tools_with_corrected_message_history",
        "model_path": str(args.model_path),
        "attention_implementation": model.config._attn_implementation,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "image_search_limit": IMAGE_SEARCH_LIMIT,
        "text_search_limit": TEXT_SEARCH_LIMIT,
        "max_rounds": args.max_rounds,
        "cases": traces,
        "all_control_flows_pass": all(trace["control_flow_pass"] for trace in traces),
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
