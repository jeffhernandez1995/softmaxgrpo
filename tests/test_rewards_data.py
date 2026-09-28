import json
import subprocess
import sys

import pytest
from datasets import Dataset

from softmaxgrpo.prepare_data import assert_disjoint, grouped_split, make_record, remove_evaluation_overlap
from softmaxgrpo.rewards import compute_score, countdown_correct, format_score, math_correct, similarity


@pytest.mark.parametrize("answer", ["<answer>(3+3)*4*1</answer>", "<answer>(3+3)*4*1=24</answer>"])
def test_countdown_correct(answer):
    assert countdown_correct(answer, [3, 3, 4, 1], 24) == 1


def test_countdown_uses_supplied_target_and_all_numbers():
    record = make_record({"nums": [10, 5, 2], "target": 52}, "countdown", "exact")
    result = compute_score(
        record["data_source"], "<answer>10*5+2</answer>", record["reward_model"]["ground_truth"]
    )
    assert result["accuracy"] == 1
    assert "52" in record["prompt"][-1]["content"]


@pytest.mark.parametrize(
    "answer",
    ["24", "(3+3)*4", "(3+3)*4*1=12", "(3+3)*4+1-1", "3**3-3", "__import__('os').system('false')", "3/(3-3)"],
)
def test_countdown_rejects_invalid_or_unsafe_expressions(answer):
    assert countdown_correct(answer, [3, 3, 4, 1], 24) == 0


def test_rewards_paper_weights_and_format():
    assert similarity("<think>unrelated reasoning</think>The cat sat.", "cat sat")["similarity"] == 1
    scores = similarity("red blue", "blue red")
    assert scores == {"similarity": 0.8, "squad_f1": 1.0, "rouge_l": 0.5}
    truth = {"task": "openthoughts", "reward": "sim", "reference": "blue red"}
    result = compute_score("unused", "<think>reason</think>red blue", json.dumps(truth))
    assert result["score"] == pytest.approx(0.35 + 0.65 * 0.8)
    assert format_score("<think></think>answer") == 0
    assert format_score("<think> </think>answer") == 0
    assert format_score("<think>reason</think><think>more</think>answer") == 0
    assert similarity("", "")["similarity"] == 0


def test_math_verify_and_similarity_are_separate():
    assert math_correct(r"\boxed{\frac{1}{2}}", "0.5") == 1
    assert math_correct(r"\boxed{2}", "3") == 0
    truth = {"task": "gsm8k", "reward": "sim", "answer": "42", "reference": "different words"}
    result = compute_score("unused", r"\boxed{42}", truth)
    assert result["accuracy"] == 1 and result["score"] == 0


@pytest.mark.parametrize(
    "task,mode,row",
    [
        ("gsm8k", "exact", {"question": "Two plus two?", "answer": "Add them. #### 4"}),
        (
            "deepmath",
            "sim",
            {
                "question": "Half?",
                "final_answer": "0.5",
                "r1_solution_1": "long answer",
                "r1_solution_2": "short",
            },
        ),
        (
            "countdown",
            "sim",
            {
                "nums": [3, 3, 4, 1],
                "target": 24,
                "messages": [{"role": "assistant", "content": "<answer>(3+3)*4*1</answer>"}],
            },
        ),
        (
            "openthoughts",
            "sim",
            {
                "conversations": [
                    {"from": "human", "value": "Explain this"},
                    {"from": "gpt", "value": "<think>reasoning</think>answer"},
                ]
            },
        ),
        ("polaris", "exact", {"problem": "How many?", "answer": "5"}),
    ],
)
def test_source_adapters(task, mode, row):
    record = make_record(row, task, mode)
    assert record["prompt"][-1]["role"] == "user"
    assert all(m["role"] != "assistant" for m in record["prompt"])
    assert json.loads(record["reward_model"]["ground_truth"])["task"] == task


def test_grouped_holdout_keeps_all_references_for_one_question_together():
    rows = [
        make_record({"question": str(i), "answer": f"reasoning {j} #### {i}"}, "gsm8k", "sim")
        for i in range(30)
        for j in range(2)
    ]
    dataset = Dataset.from_list(rows)
    train, validation = grouped_split(dataset, 3, 42)
    assert len(validation) == 6
    assert_disjoint(train, validation)
    _, shuffled_validation = grouped_split(dataset.shuffle(seed=7), 3, 42)
    assert {r["id"] for r in validation["extra_info"]} == {r["id"] for r in shuffled_validation["extra_info"]}


def test_prepare_cli_round_trip(tmp_path):
    source = tmp_path / "source.parquet"
    Dataset.from_list(
        [{"question": f"Question {i}", "answer": f"Work #### {i}"} for i in range(30)]
    ).to_parquet(source)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "softmaxgrpo.prepare_data",
            "--task",
            "gsm8k",
            "--train-file",
            str(source),
            "--output-dir",
            str(tmp_path / "data"),
        ],
        check=True,
    )
    folder = tmp_path / "data/gsm8k_exact"
    train = Dataset.from_parquet(str(folder / "train.parquet"))
    validation = Dataset.from_parquet(str(folder / "validation.parquet"))
    assert len(train) == 27 and len(validation) == 3
    assert_disjoint(train, validation)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["train_rows"] == 27
    assert str(tmp_path) not in (folder / "manifest.json").read_text()


def test_qwen_evaluation_prompts_removed_from_training():
    train = Dataset.from_list(
        [make_record({"problem": p, "answer": "5"}, "polaris", "exact") for p in ("shared", "training-only")]
    )
    val = Dataset.from_list([make_record({"problem": "shared", "answer": "5"}, "math500", "exact")])
    clean = remove_evaluation_overlap(train, [val])
    assert len(clean) == 1
    assert_disjoint(clean, val)


def test_small_dataset_rejected_before_gpu_startup(tmp_path):
    from omegaconf import OmegaConf

    from softmaxgrpo.train import check_data

    file = tmp_path / "tiny.parquet"
    Dataset.from_list([{"value": 1}]).to_parquet(file)
    cfg = OmegaConf.create(
        {"data": {"train_files": str(file), "val_files": str(file), "train_batch_size": 64}}
    )
    with pytest.raises(ValueError, match="at least 64"):
        check_data(cfg)
