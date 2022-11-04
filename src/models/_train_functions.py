from __future__ import annotations

import torch
from tqdm import tqdm, trange

from ..visualization.tensorboard import add_sample_predictions
from ._train_config import TrainConfig


def run_epoch(
    model: torch.nn.Module,
    train_config: TrainConfig,
    mode: str,
    current_epoch: int,
    profile: bool = False,
) -> tuple[float, float]:
    model.to(train_config.device)

    if mode == "train":
        model.train()
        loader = train_config.train_loader
    elif mode == "eval":
        model.eval()
        loader = train_config.val_loader
    else:
        raise ValueError(f'"{mode}" is not a valid mode. Mode must be one of: ["train", "eval"]')

    epoch_loss = 0.0
    epoch_accuracy = 0.0

    profiler = None
    wait = 1
    warmup = 1
    active = 5
    repeat = 2

    if profile and train_config.device == "cuda":
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(train_config.writer.log_dir),
            record_shapes=True,
            profile_memory=True,
        )
        profiler.start()

    with torch.set_grad_enabled(mode == "train"):
        for i, (X_batch, y_batch) in tqdm(
            enumerate(loader),
            total=len(loader),
            leave=False,
            desc=f"{mode.capitalize()} batch",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            X_batch = X_batch.to(train_config.device, non_blocking=True)
            y_batch = y_batch.to(train_config.device, non_blocking=True)

            logits = model(X_batch)
            y_pred = torch.argmax(logits, dim=-1)

            loss = train_config.criterion(logits, y_batch)
            accuracy = (y_pred == y_batch).float().mean()

            if mode == "train":
                loss.backward()
                train_config.optimizer.step()
                train_config.optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() / len(X_batch)
            epoch_accuracy += accuracy.item()

            if profiler:
                if i >= (wait + warmup + active) * repeat:
                    profiler.stop()
                profiler.step()

    if mode == "eval":
        add_sample_predictions(
            train_config.writer, X_batch[:4], logits[:4], y_batch[:4], train_config.classes, current_epoch
        )

    epoch_loss /= len(loader)
    epoch_accuracy /= len(loader)

    return epoch_loss, epoch_accuracy


def train(model: torch.nn.Module, train_config: TrainConfig) -> None:

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(train_config.optimizer, verbose=True)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(
        train_config.optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05
    )

    for epoch in trange(
        1,
        train_config.total_epochs + 1,
        leave=True,
        desc="Epoch",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        profile = epoch == 2
        train_loss, train_accuracy = run_epoch(
            model, train_config, mode="train", current_epoch=epoch, profile=profile
        )

        if epoch > train_config.swa_start:
            val_loss, val_accuracy = run_epoch(swa_model, train_config, mode="eval", current_epoch=epoch)
        else:
            val_loss, val_accuracy = run_epoch(model, train_config, mode="eval", current_epoch=epoch)

        if epoch > train_config.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(val_loss)

        train_config.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, global_step=epoch)
        train_config.writer.add_scalars(
            "Accuracy", {"train": train_accuracy, "val": val_accuracy}, global_step=epoch
        )
        train_config.writer.flush()

    torch.optim.swa_utils.update_bn(train_config.train_loader, swa_model)
