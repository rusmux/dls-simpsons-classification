from __future__ import annotations

import io
import logging
import os
import random
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image
from sklearn.model_selection._split import _validate_shuffle_split


def get_logger() -> logging.Logger:
    log_format = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)
    return logger


def seed_everything(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)


def plotly_fig_to_array(fig: go.Figure) -> np.ndarray:
    fig_bytes = fig.to_image(format="png")
    buffer = io.BytesIO(fig_bytes)
    image = Image.open(buffer)
    array = np.asarray(image) / 255  # type: ignore
    return array


def get_train_val_size(
    n_samples: int, train_size: Optional[float] = None, val_size: Optional[float] = None
) -> tuple[float, float]:
    _validate_shuffle_split(n_samples, train_size, val_size, default_test_size=0.25)
    if train_size is not None:
        return train_size, 1 - train_size
    if val_size is not None:
        return 1 - val_size, val_size
    raise ValueError("either train_size or val_size must be provided")
