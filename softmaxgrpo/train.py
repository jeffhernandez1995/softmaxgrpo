"""A small entry point using upstream verl's public TaskRunner extension hook."""

import os
from importlib.metadata import version
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict


def check_config(config):
    if config.algorithm.adv_estimator != "softmaxgrpo":
        raise ValueError("This release supports algorithm.adv_estimator=softmaxgrpo only")
    import math

    tau = float(config.algorithm.softmaxgrpo_tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("algorithm.softmaxgrpo_tau must be finite and positive")
    if config.actor_rollout_ref.rollout.n < 2:
        raise ValueError("SoftmaxGRPO training needs at least two rollouts per prompt")
    if config.algorithm.use_kl_in_reward:
        raise ValueError("Use the reference KL loss; keep outcome rewards separate from KL")


def check_data(config):
    """Catch missing files and undersized datasets before initializing GPU workers."""
    import pyarrow.parquet as pq

    for split, files in (("train", config.data.train_files), ("validation", config.data.val_files)):
        files = [files] if isinstance(files, str) else list(files)
        count = sum(pq.read_metadata(Path(file).expanduser()).num_rows for file in files)
        minimum = config.data.train_batch_size if split == "train" else 1
        if count < minimum:
            raise ValueError(
                f"{split} has {count} rows, but this configuration needs at least {minimum}. "
                "Provide the full experiment dataset or explicitly reduce data.train_batch_size."
            )


@hydra.main(config_path="configs", config_name="gsm8k", version_base=None)
def main(config):
    check_config(config)
    check_data(config)
    if version("verl") != "0.6.1":
        raise RuntimeError("Install the pinned verl==0.6.1 dependency from this release")
    # Resolve the installed reward file, without depending on the checkout location.
    config.custom_reward_function.path = str(Path(__file__).with_name("rewards.py"))
    OmegaConf.resolve(config)

    import ray
    from verl.trainer.main_ppo import TaskRunner, run_ppo

    class SoftmaxGRPOTaskRunner(TaskRunner):
        def run(self, worker_config):
            # Registration must happen in the Ray process that computes advantages.
            from softmaxgrpo.advantage import register_with_verl

            register_with_verl()
            return super().run(worker_config)

    # A Slurm launcher sets this after starting the allocation's Ray cluster.
    if os.environ.get("RAY_ADDRESS"):
        with open_dict(config.ray_kwargs.ray_init):
            config.ray_kwargs.ray_init.address = os.environ["RAY_ADDRESS"]
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SoftmaxGRPOTaskRunner))


if __name__ == "__main__":
    main()
