#!/usr/bin/env python3
"""Run a deterministic one-round MMSearch-R1 checkpoint smoke test."""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import re
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MAX_PIXELS = 672 * 672


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--expected-answer")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    prompt_path = Path("mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl")
    with prompt_path.open("rb") as handle:
        round_1_prompt = pickle.load(handle).replace("<image>", "").strip()

    encoded_image = base64.b64encode(args.image.read_bytes()).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{round_1_prompt}\nQuestion: {args.question}\nImage: ",
                },
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{encoded_image}",
                    "max_pixels": MAX_PIXELS,
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
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        use_fast=False,
    )
    load_seconds = time.monotonic() - load_started
    print("[PASS] model load")

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
    print("[PASS] image processor")

    generation_started = time.monotonic()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    generation_seconds = time.monotonic() - generation_started

    generated_only = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print("[PASS] generation")
    print(response)

    reason_match = re.search(r"<reason>(.*?)</reason>", response, flags=re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    has_reason = reason_match is not None
    has_answer = answer_match is not None
    parsed_answer = answer_match.group(1).strip() if answer_match else None
    exact_match = None
    if args.expected_answer is not None and parsed_answer is not None:
        exact_match = normalize_answer(parsed_answer) == normalize_answer(args.expected_answer)

    result = {
        "model_path": str(args.model_path),
        "image": str(args.image),
        "question": args.question,
        "expected_answer": args.expected_answer,
        "response": response,
        "parsed_answer": parsed_answer,
        "has_reason": has_reason,
        "has_answer": has_answer,
        "exact_match": exact_match,
        "attention_implementation": model.config._attn_implementation,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "input_tokens": int(inputs.input_ids.shape[-1]),
        "output_tokens": int(generated_only[0].shape[-1]),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if has_reason:
        print("[PASS] <reason>...</reason>")
    else:
        print("[FAIL] missing <reason>...</reason>")
    if has_answer:
        print("[PASS] <answer>...</answer>")
    else:
        print("[FAIL] missing <answer>...</answer>")
    if exact_match is not None:
        print(f"[{'PASS' if exact_match else 'WARN'}] exact match: {exact_match}")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not (has_reason and has_answer):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
