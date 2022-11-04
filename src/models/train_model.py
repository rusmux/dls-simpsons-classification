from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from src.data import get_dataset, get_train_val_loader
from src.utils import get_logger, get_train_val_size, seed_everything
from src.visualization.tensorboard import add_embeddings, add_model, add_sample_images

from ._train_config import TrainConfig
from ._train_functions import train
from .custom_models import AlexNet


@click.command()
@click.option("--batch_size", "-b", required=True, type=int)
@click.option("--epochs", "-e", required=True, type=int)
@click.option("--device", "-d", required=True, type=str)
@click.option("--learning_rate", "-lr", default=1e-4, type=float)
@click.option("--train_size", "-t", default=None, type=float)
@click.option("--val_size", "-v", default=None, type=float)
@click.option("--swa_start", "-swa", default=None, type=int)
@click.option("--weighted", "-w", default=False, type=bool)
@click.option("--random_seed", "-r", default=0, type=int)
@click.option("--experiment_suffix", "-s", default="", type=str)
def main(
    batch_size: int,
    epochs: int,
    device: str,
    learning_rate: float = 1e-4,
    train_size: Optional[float] = None,
    val_size: Optional[float] = None,
    swa_start: Optional[int] = None,
    weighted: bool = False,
    random_seed: int = 0,
    experiment_suffix: str = "",
):
    seed_everything(random_seed)

    logger = get_logger()
    project_dir = Path(__file__).resolve().parents[2]

    log_dir = "runs/dls-simpsons-classification"
    if experiment_suffix:
        log_dir += "-" + experiment_suffix
    writer = SummaryWriter(log_dir)
    logger.info(f'started tensorboard writer in "{log_dir}"')

    train_dataset = get_dataset(project_dir / "data/raw/train")  # type: ignore
    classes = np.array(train_dataset.classes)

    add_sample_images(writer, train_dataset)
    add_embeddings(writer, train_dataset)

    if train_size is None and val_size is None:
        val_size = 0.2
    train_size, val_size = get_train_val_size(len(train_dataset), train_size, val_size)

    train_loader, val_loader = get_train_val_loader(
        train_dataset, batch_size, train_size, val_size, weighted=weighted
    )

    model = AlexNet()
    logger.info("the model to be used for training:")
    model_summary = summary(model, input_size=(batch_size, 3, 227, 227), depth=2, device="cpu", verbose=0)
    print(model_summary)
    add_model(writer, model, next(iter(train_loader))[0][:4])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if swa_start is None:
        swa_start = epochs

    logger.info(
        f"running {epochs} training epochs "
        f"on {device} device, "
        f"batch size {batch_size}, "
        f"train size {train_size}, "
        f"validation size {val_size}, "
        f"swa start {swa_start}\n"
    )

    train_config = TrainConfig(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        total_epochs=epochs,
        swa_start=swa_start,
        device=device,
        classes=classes,
        writer=writer,
    )

    train(model, train_config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
