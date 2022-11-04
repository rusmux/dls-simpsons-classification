from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Extra, conint, validator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class TrainConfig(BaseModel):
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    total_epochs: conint(strict=True, gt=0)  # type: ignore
    swa_start: conint(strict=True, gt=0)  # type: ignore
    device: str
    classes: np.ndarray
    writer: SummaryWriter

    @classmethod
    @validator("swa_start")
    def check_swa_start(cls, swa_start: int, values: dict[str, Any]) -> int:
        if swa_start > values["total_epochs"]:
            raise ValueError("swa_start must be less or equal to total_epochs")
        return swa_start

    @classmethod
    @validator("device")
    def check_device(cls, device: str) -> str:
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise ValueError("torch not compiled with CUDA enabled")
        if device == "mps":
            if torch.has_mps:
                return "mps"
            raise ValueError("this machine does not have mps")
        raise ValueError(
            f'"{device}" is not a valid device. Try using "cuda" if you have a GPU or '
            f'"mps" if you have an Apple M-series chip. Use "cpu" otherwise.'
        )

    class Config:  # pylint: disable=too-few-public-methods
        arbitrary_types_allowed = True
        extra = Extra.forbid
        validate_assignment = True
