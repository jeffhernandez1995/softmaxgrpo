# OpenThoughts transfer evaluation

Export the final OpenThoughts actor first:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir outputs/openthoughts_sim/global_step_3220/actor \
  --target_dir outputs/openthoughts_hf
```

The examples below specify an evaluation protocol explicitly. Record the tool versions,
dataset revisions, prompts, decoding settings, and exported checkpoint with each result.
Training completion alone does not verify the paper's reported scores.

## MMLU and GPQA

Use the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).
Install it in a separate evaluation environment to keep training dependencies fixed.
The examples use five-shot MMLU and zero-shot GPQA main with multiple-choice likelihood
scoring. GPQA requires access to the dataset through Hugging Face.

```bash
python -m pip install 'lm-eval==0.4.9.1'
lm_eval --model hf --model_args pretrained=outputs/openthoughts_hf,dtype=bfloat16 \
  --tasks mmlu --num_fewshot 5 --batch_size auto --device cuda \
  --output_path outputs/transfer/mmlu --log_samples
lm_eval --model hf --model_args pretrained=outputs/openthoughts_hf,dtype=bfloat16 \
  --tasks gpqa_main --num_fewshot 0 --batch_size auto --device cuda \
  --output_path outputs/transfer/gpqa --log_samples
```

## AlpacaEval 2.0

Obtain the official evaluation instructions from the
[AlpacaEval dataset](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)
as a JSON list containing `instruction`. Generate responses locally:

```bash
python -m softmaxgrpo.evaluate --model outputs/openthoughts_hf \
  --instructions data/alpaca_eval.json --max-tokens 6144 --max-model-len 16384 \
  --output outputs/transfer/alpaca_outputs.json
```

Then run the [official evaluator](https://github.com/tatsu-lab/alpaca_eval) in its own
environment. This step sends the generated answers to the configured judge provider
and incurs API usage; it is separate from local generation.

```bash
python -m pip install 'alpaca-eval==0.6.6'
export IS_ALPACA_EVAL_2=True
# Set OPENAI_API_KEY in your environment before invoking the evaluator.
alpaca_eval --model_outputs outputs/transfer/alpaca_outputs.json \
  --annotators_config weighted_alpaca_eval_gpt4_turbo \
  --output_path outputs/transfer/alpaca_eval
```

Report the length-controlled win rate. Store the evaluator's reference-output version
and annotation cache with the results.
