"""Generate and score a held-out Parquet set, or generate AlpacaEval instruction outputs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        source = row["data_source"]
        for metric in row["metrics"][0]:
            grouped[source][metric].append(sum(m[metric] for m in row["metrics"]) / len(row["metrics"]))
        if "accuracy" in row["metrics"][0]:
            grouped[source][f"pass@{len(row['metrics'])}"].append(
                float(any(m["accuracy"] == 1 for m in row["metrics"]))
            )
    return {
        source: {name: sum(values) / len(values) for name, values in metrics.items()}
        for source, metrics in grouped.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exported Hugging Face checkpoint")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path, nargs="+", help="Prepared validation Parquet files")
    inputs.add_argument("--instructions", type=Path, help="AlpacaEval JSON list with an instruction field")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1 or (args.samples > 1 and args.temperature == 0):
        parser.error("Use positive temperature with multiple samples, or one greedy sample")
    if args.instructions and args.samples != 1:
        parser.error("AlpacaEval expects one answer per instruction")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from softmaxgrpo.rewards import compute_score, final_text

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.data:
        rows = list(load_dataset("parquet", data_files=[str(p) for p in args.data], split="train"))
    else:
        instructions = json.loads(args.instructions.read_text())
        rows = [
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Put your reasoning in <think> "
                        "</think> tags, followed by your final answer.",
                    },
                    {"role": "user", "content": row["instruction"]},
                ],
                "instruction": row["instruction"],
            }
            for row in instructions
        ]
    if not rows:
        parser.error("The evaluation set is empty")
    prompts = [
        tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        for row in rows
    ]
    if max(map(len, prompts)) + args.max_tokens > args.max_model_len:
        parser.error("Increase --max-model-len; evaluation prompts are never silently truncated")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        SamplingParams(
            n=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )
    records = []
    for row, output in zip(rows, outputs):
        completions = [o.text for o in output.outputs]
        if args.instructions:
            records.append(
                {
                    "instruction": row["instruction"],
                    "output": final_text(completions[0]),
                    "generator": "softmaxgrpo",
                }
            )
        else:
            metrics = [
                compute_score(row["data_source"], text, row["reward_model"]["ground_truth"])
                for text in completions
            ]
            records.append(
                {
                    "id": row["extra_info"]["id"],
                    "data_source": row["data_source"],
                    "responses": completions,
                    "metrics": metrics,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    report["config"]["data"] = [str(p) for p in args.data] if args.data else None
    if args.data:
        report["metrics"] = summarize(records)
        print(json.dumps(report["metrics"], indent=2))
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
