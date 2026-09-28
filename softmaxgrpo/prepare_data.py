"""Prepare verl Parquet files. No training code or local paths from the old runner."""

import argparse
import hashlib
import json
from pathlib import Path

SYSTEM = "You are a helpful assistant. Think through the problem before providing the answer."
MATH_SUFFIX = r"Please reason step by step, and put your final answer within \boxed{}."
SOURCES = {
    "gsm8k": ("openai/gsm8k", "main"),
    "countdown": ("Jiayi-Pan/Countdown-Tasks-3to4", None),
    "countdown_sim": ("HuggingFaceTB/Countdown-Task-GOLD", "verified_Qwen2.5-7B-Instruct"),
    "deepmath": ("zwhe99/DeepMath-103K", None),
    "openthoughts": ("open-thoughts/OpenThoughts3-1.2M", None),
    "polaris": ("POLARIS-Project/Polaris-Dataset-53K", None),
    "aime25": ("math-ai/aime25", None),
    "math500": ("HuggingFaceH4/MATH-500", None),
}


def first_text(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Missing nonempty field; expected one of {keys}")


def conversation(row):
    messages = row.get("messages", row.get("conversations", []))
    if isinstance(messages, str):
        messages = json.loads(messages)
    roles = {"human": "user", "gpt": "assistant"}
    return [
        {
            "role": roles.get(m.get("role", m.get("from")), m.get("role", m.get("from"))),
            "content": str(m.get("content", m.get("value", ""))),
        }
        for m in messages
    ]


def make_record(row, task, reward):
    from softmaxgrpo.rewards import final_text

    truth = {"task": task, "reward": reward}
    system = SYSTEM
    if task == "gsm8k":
        question = first_text(row, "question")
        reference = first_text(row, "answer")
        truth.update(answer=reference.rsplit("####", 1)[-1].strip(), reference=reference)
        prompt = question + "\n" + MATH_SUFFIX
    elif task == "deepmath":
        question = first_text(row, "question")
        answer = first_text(row, "final_answer")
        solutions = [
            final_text(row[k])
            for k in ("r1_solution_1", "r1_solution_2", "r1_solution_3")
            if row.get(k) and final_text(row[k])
        ]
        if reward == "sim" and not solutions:
            raise ValueError("DeepMath similarity reward requires a verified solution")
        truth.update(answer=answer, reference=min(solutions, key=len) if solutions else answer)
        prompt = question + "\n" + MATH_SUFFIX
    elif task == "countdown":
        numbers = [int(n) for n in row["nums"]]
        target = int(row["target"])
        references = [m["content"] for m in conversation(row) if m["role"] == "assistant"]
        reference = references[-1] if references else row.get("response", row.get("completion", ""))
        truth.update(numbers=numbers, target=target, reference=str(reference))
        prompt = (
            f"Using the numbers {numbers}, create an equation that equals {target}. "
            "Use only +, -, *, / and parentheses, and use each number exactly once. "
            "Show your work in <think> </think> tags and return the final expression in "
            "<answer> </answer> tags."
        )
    elif task == "openthoughts":
        messages = conversation(row)
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("Expected one human/assistant pair in OpenThoughts")
        prompt = messages[0]["content"]
        system += " Put your reasoning in <think> </think> tags, followed by your final answer."
        truth["reference"] = messages[1]["content"]
    elif task in {"polaris", "aime25", "math500"}:
        prompt = first_text(row, "problem") + "\n" + MATH_SUFFIX
        truth["answer"] = first_text(row, "answer")
        system = None  # Match the inherited Qwen3 user-only prompts.
    else:
        raise ValueError(f"Unsupported task: {task}")
    if reward == "sim" and not truth.get("reference", "").strip():
        raise ValueError(f"{task} similarity training requires a nonempty reference")
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    # Group duplicate questions before creating a holdout (OpenThoughts has repeated questions).
    uid = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    return {
        "data_source": f"softmaxgrpo/{task}/{reward}",
        "prompt": messages,
        "reward_model": {"style": "rule", "ground_truth": json.dumps(truth, ensure_ascii=False)},
        "extra_info": {"id": uid, "task": task, "reward": reward},
    }


def grouped_split(dataset, size, seed):
    """Deterministic holdout of distinct prompts, independent of source row order."""
    uids = {row["id"] for row in dataset["extra_info"]}
    if len(uids) < 2:
        raise ValueError("At least two distinct prompts are needed for a holdout")
    count = min(size, max(1, len(uids) // 10))
    ordered = sorted(uids, key=lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest())
    heldout = set(ordered[:count])
    train, val = [], []
    for i, row in enumerate(dataset["extra_info"]):
        (val if row["id"] in heldout else train).append(i)
    return dataset.select(train), dataset.select(val)


def assert_disjoint(train, validation):
    train_ids = {x["id"] for x in train["extra_info"]}
    val_ids = {x["id"] for x in validation["extra_info"]}
    if train_ids & val_ids:
        raise ValueError("Training and validation contain overlapping prompts")


def remove_evaluation_overlap(train, evaluations):
    evaluation_ids = {row["id"] for dataset in evaluations for row in dataset["extra_info"]}
    return train.filter(lambda row: row["extra_info"]["id"] not in evaluation_ids)


def source_dataset(task, revision=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    name, subset = SOURCES[task]
    sha = HfApi().dataset_info(name, revision=revision).sha
    return load_dataset(name, subset, revision=sha), {"dataset": name, "subset": subset, "revision": sha}


def convert(dataset, task, reward):
    return dataset.map(lambda row: make_record(row, task, reward), remove_columns=dataset.column_names)


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, choices=["gsm8k", "countdown", "deepmath", "openthoughts", "qwen3"]
    )
    parser.add_argument("--reward", choices=["exact", "sim"])
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--revision", help="HF dataset revision; resolved to a commit in the manifest")
    parser.add_argument("--train-file", type=Path, help="Optional local source Parquet")
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.validation_size < 1:
        parser.error("--validation-size must be positive")
    if args.task == "qwen3":
        if args.reward == "sim" or args.train_file or args.validation_file:
            parser.error("qwen3 uses the Polaris training set and separate exact-reward math evaluations")
        out = args.output_dir / "qwen3"
        out.mkdir(parents=True, exist_ok=True)
        sources = {}
        prepared = {}
        for task, split, filename in [
            ("polaris", "train", "train"),
            ("aime25", "test", "aime25"),
            ("math500", "test", "math500"),
        ]:
            raw, source = source_dataset(task, args.revision if task == "polaris" else None)
            converted = convert(raw[split], task, "exact")
            prepared[filename] = converted
            sources[task] = {**source, "split": split, "rows": len(converted)}
        train_size = len(prepared["train"])
        prepared["train"] = remove_evaluation_overlap(
            prepared["train"], [prepared["aime25"], prepared["math500"]]
        )
        sources["polaris"]["evaluation_overlaps_removed"] = train_size - len(prepared["train"])
        sources["polaris"]["rows"] = len(prepared["train"])
        for filename, converted in prepared.items():
            converted.to_parquet(str(out / f"{filename}.parquet"))
        (out / "manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
        return

    mode = args.reward or ("exact" if args.task in {"gsm8k", "countdown", "deepmath"} else "sim")
    if args.task == "openthoughts" and mode != "sim":
        parser.error(f"{args.task} uses similarity rewards")
    if args.validation_file and not args.train_file:
        parser.error("--validation-file requires --train-file")
    if args.train_file:
        files = {"train": str(args.train_file)}
        if args.validation_file:
            files["validation"] = str(args.validation_file)
        raw = load_dataset("parquet", data_files=files)
        # Include content hashes, not identifying absolute source paths.
        hashes = {}
        for split, file in files.items():
            digest = hashlib.sha256()
            with open(file, "rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[split] = digest.hexdigest()
        source = {"local_sha256": hashes}
    else:
        source_key = "countdown_sim" if args.task == "countdown" and mode == "sim" else args.task
        raw, source = source_dataset(source_key, args.revision)
    if not len(raw["train"]):
        raise ValueError(
            "The source has no training rows after filtering; provide the experiment's original files"
        )
    train = convert(raw["train"], args.task, mode)
    eval_split = next((key for key in ("validation", "test") if key in raw), None)
    if eval_split:
        validation = convert(raw[eval_split], args.task, mode)
        split_description = f"source train / {eval_split}"
    else:
        train, validation = grouped_split(train, args.validation_size, args.seed)
        split_description = "hash-ranked distinct prompt holdout, at most 10% of unique prompts"
    if not len(validation):
        raise ValueError("Validation must be nonempty")
    assert_disjoint(train, validation)
    out = args.output_dir / f"{args.task}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(str(out / "train.parquet"))
    validation.to_parquet(str(out / "validation.parquet"))
    manifest = {
        **source,
        "task": args.task,
        "reward": mode,
        "seed": args.seed,
        "split_policy": split_description,
        "requested_validation_prompts": args.validation_size,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }
    if args.task == "countdown":
        manifest["countdown_variant"] = "full source; supplied numbers and targets"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
