"""ResNet-50 ImageNet training with sampled-label SoftmaxGRPO rewards only."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from softmaxgrpo.advantage import softmax_advantages


def sampled_loss(logits, labels, rollouts=1024, tau=0.2):
    """One rollout = one class sampled from the current model; reward is correctness."""
    if rollouts < 2:
        raise ValueError("At least two rollouts per image are required")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    with torch.no_grad():
        actions = torch.multinomial(log_probs.exp(), rollouts, replacement=True)
        rewards = actions.eq(labels[:, None]).float()
        advantages = softmax_advantages(rewards, tau)
    return -(log_probs.gather(1, actions) * advantages).mean(), rewards.mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images).float()
        probabilities = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1).double()
        # Exact probability of at least one correct label in k independent samples.
        pass128 = -torch.expm1(128 * torch.log1p(-probabilities))
        totals += torch.stack((logits.argmax(-1).eq(labels).sum(), probabilities.sum(), pass128.sum()))
        count += len(labels)
    if count == 0:
        raise ValueError("ImageNet validation data is empty")
    return dict(zip(("top1", "pass@1", "pass@128"), (totals / count).tolist()))


def loaders(args):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train = datasets.ImageFolder(args.data_dir / "train", transform=train_transform)
    validation = datasets.ImageFolder(args.data_dir / "val", transform=val_transform)
    if len(train.classes) != 1000 or train.class_to_idx != validation.class_to_idx:
        raise ValueError("ImageNet train/ and val/ must have the same 1,000 class subdirectories")
    kwargs = {"num_workers": args.workers, "pin_memory": args.device.startswith("cuda")}
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs),
        DataLoader(validation, batch_size=args.eval_batch_size, shuffle=False, **kwargs),
    )


def main():
    from torchvision.models import resnet50

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imagenet"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, help="Resume an epoch-boundary checkpoint")
    args = parser.parse_args()
    if not math.isfinite(args.tau) or args.tau <= 0 or args.rollouts < 2:
        parser.error("--tau must be finite and positive and --rollouts must be >= 2")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.eval_every) < 1:
        parser.error("Epoch, batch, and evaluation counts must be positive")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup-epochs must be between zero and epochs - 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train, validation = loaders(args)
    model = resnet50(weights=None).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total_steps, warmup = args.epochs * len(train), args.warmup_epochs * len(train)

    def schedule(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch, step, best = 0, 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ("tau", "rollouts", "batch_size", "epochs", "lr", "warmup_epochs"):
            if state["config"][key] != getattr(args, key):
                raise ValueError(f"Resume requires the same {key} as the checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, step, best = state["next_epoch"], state["step"], state["best_pass1"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def log(record):
        line = json.dumps({"step": step, **record})
        print(line, flush=True)
        with (args.output_dir / "metrics.jsonl").open("a") as file:
            file.write(line + "\n")

    if not args.resume:
        log({"validation": evaluate(model, validation, device)})
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for images, labels in train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, reward = sampled_loss(model(images), labels, args.rollouts, args.tau)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0:
                log({"loss": loss.item(), "reward": reward.item(), "lr": scheduler.get_last_lr()[0]})
            if step % args.eval_every == 0:
                log({"validation": evaluate(model, validation, device)})
                model.train()
        metrics = evaluate(model, validation, device)
        log({"epoch": epoch + 1, "validation": metrics})
        improved = metrics["pass@1"] > best
        best = max(best, metrics["pass@1"])
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "next_epoch": epoch + 1,
            "step": step,
            "best_pass1": best,
            "config": config,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        torch.save(state, args.output_dir / "latest.pt")
        if improved:
            torch.save(state, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
