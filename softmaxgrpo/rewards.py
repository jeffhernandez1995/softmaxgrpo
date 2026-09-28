"""Task rewards in verl's custom_reward_function interface.
"""

import ast
import json
import re
import string
from collections import Counter
from fractions import Fraction


def final_text(text):
    """Remove an optional thinking trace and final-answer wrapper."""
    text = str(text).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    answers = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return (answers[-1] if answers else text).strip()


def normalized_tokens(text):
    text = final_text(text).lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def similarity(prediction, reference):
    """Word-level overlap after SQuAD normalization; ROUGE-L uses the same tokens."""
    from rapidfuzz.distance import LCSseq

    pred, gold = normalized_tokens(prediction), normalized_tokens(reference)
    if not pred or not gold:
        return {"similarity": 0.0, "squad_f1": 0.0, "rouge_l": 0.0}
    denominator = len(pred) + len(gold)
    f1 = 2.0 * sum((Counter(pred) & Counter(gold)).values()) / denominator
    rouge = 2.0 * LCSseq.similarity(pred, gold) / denominator
    return {"similarity": 0.6 * f1 + 0.4 * rouge, "squad_f1": f1, "rouge_l": rouge}


def format_score(text):
    # A nonempty think block followed by a nonempty answer, with no extra think tags.
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return 0.0
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*(.*?)\s*", text, re.DOTALL)
    return float(bool(match and match.group(1).strip() and match.group(2).strip()))


def math_correct(text, gold):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    # Gold answers are plain expressions or LaTeX; wrapping supplies an unambiguous target.
    expected = parse(r"\boxed{" + str(gold) + "}", extraction_config=[LatexExtractionConfig()])
    actual = parse(final_text(text), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    return float(bool(expected and actual and verify(expected, actual)))


def countdown_correct(text, numbers, target):
    """Evaluate only a bounded arithmetic AST; never execute model-generated code."""
    expression = final_text(text).replace("×", "*").replace("÷", "/").replace("−", "-")
    boxed = re.search(r"\\boxed\{([^{}]*)\}", expression)
    if boxed:
        expression = boxed.group(1)
    expression = expression.strip().strip("`$").strip().rstrip(".")
    if len(expression) > 256:
        return 0.0
    parts = expression.split("=")
    if len(parts) > 2:
        return 0.0
    if len(parts) == 2:
        try:
            if Fraction(parts[1].strip()) != Fraction(int(target)):
                return 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    used = []

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 10**6:
            used.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("Only supplied integers, parentheses and + - * / are allowed")

    try:
        value = evaluate(ast.parse(parts[0].strip(), mode="eval").body)
        return float(Counter(used) == Counter(int(n) for n in numbers) and value == int(target))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, RecursionError):
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return score plus separate metrics; validation accuracy is not similarity."""
    truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    task, mode = truth["task"], truth["reward"]
    metrics = {}
    if task == "countdown":
        metrics["accuracy"] = countdown_correct(solution_str, truth["numbers"], truth["target"])
    elif task in {"gsm8k", "deepmath", "polaris", "aime25", "math500"}:
        metrics["accuracy"] = math_correct(solution_str, truth["answer"])
    if mode == "exact":
        return {"score": metrics["accuracy"], **metrics}
    if mode != "sim" or not truth.get("reference"):
        raise ValueError("Similarity training requires a nonempty reference and reward='sim'")
    metrics.update(similarity(solution_str, truth["reference"]))
    score = metrics["similarity"]
    if task == "openthoughts":
        metrics["format"] = format_score(solution_str)
        score = 0.35 * metrics["format"] + 0.65 * score
    return {"score": score, **metrics}
