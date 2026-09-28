from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from softmaxgrpo.train import check_config


@pytest.mark.parametrize(
    "name",
    [
        "gsm8k",
        "gsm8k_sim",
        "countdown",
        "countdown_sim",
        "deepmath",
        "deepmath_sim",
        "openthoughts",
        "qwen3",
        "qwen3_4b",
    ],
)
def test_configs_compose_against_upstream_verl(name):
    pytest.importorskip("verl")
    config_dir = Path(__file__).resolve().parents[1] / "softmaxgrpo/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=name)
        OmegaConf.resolve(cfg)
    check_config(cfg)
    assert cfg.critic.enable is False
    assert cfg.algorithm.adv_estimator == "softmaxgrpo"
    assert (
        cfg.trainer.total_training_steps is None
        or cfg.trainer.total_epochs >= cfg.trainer.total_training_steps
    )


def test_shell_syntax():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    for pattern in ("scripts/**/*.sh", "scripts/**/*.sbatch"):
        for path in root.glob(pattern):
            subprocess.run(["bash", "-n", str(path)], check=True)
