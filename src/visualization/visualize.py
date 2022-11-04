import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from custom_plotly_templates import set_custom_show_config, set_custom_template

from src.utils import get_logger

project_dir = Path(__file__).resolve().parents[2]
logger = get_logger()


def get_images_per_class_distribution(class_lengths: pd.DataFrame) -> go.Figure:
    fig = px.histogram(class_lengths["images"], nbins=100, histnorm="")

    fig.update_layout(
        title="Распределение количества изображений на класс",
        xaxis_title=None,
        xaxis_showgrid=True,
        showlegend=False,
    )

    return fig


def get_images_per_class_cdf(class_lengths: pd.DataFrame) -> go.Figure:
    fig = px.ecdf(class_lengths["images"])

    fig.update_layout(
        title="CDF количества изображений на класс",
        xaxis_title=None,
        yaxis_title=None,
        yaxis_tickformat="p",
        showlegend=False,
    )

    return fig


def get_top_classes_by_images(class_lengths: pd.DataFrame, n_top: int, top_dir: str = "max") -> go.Figure:
    if top_dir == "max":
        top_classes = class_lengths.head(n_top)
        top_dir_ru = "наибольшим"
    elif top_dir == "min":
        top_classes = class_lengths.tail(n_top)
        top_dir_ru = "наименьшим"
    else:
        raise ValueError(f'{top_dir} is not a valid option. Must be one of: ["max", "min"]')

    fig = px.bar(
        x=top_classes["images"], y=top_classes["character"].str.replace("_", " ").str.title(), text_auto=True
    )

    fig.update_layout(
        title=f"Топ-{n_top} классов с {top_dir_ru} количеством изображений",
        xaxis_title=None,
        yaxis_title=None,
        xaxis_showgrid=False,
        xaxis_showticklabels=False,
        yaxis_linecolor="black",
        yaxis_categoryorder="total ascending",
    )

    fig.write_image(project_dir / f"reports/figures/top_classes_{top_dir}.png", scale=5)
    logger.info("figure added to /reports/figures")

    return fig


def get_image_dims_distribution(dims: Any, dim_names_ru: str) -> go.Figure:
    fig = px.histogram(dims, marginal="box")

    fig.update_layout(
        title=f"Распределение {dim_names_ru} изображений",
        xaxis_title=None,
        yaxis_title=None,
        xaxis_showgrid=True,
        yaxis_tickformat="p",
        showlegend=False,
    )

    return fig


def get_image_aspects_distribution(lengths: Iterable, heights: Iterable) -> go.Figure:
    aspect_ratio = torch.Tensor(lengths) / torch.Tensor(heights)

    fig = px.histogram(aspect_ratio, marginal="box")

    fig.update_layout(
        title="Распределение аспекта изображений",
        xaxis_title=None,
        yaxis_title=None,
        xaxis_showgrid=True,
        yaxis_tickformat="p",
        showlegend=False,
    )

    return fig


def main() -> None:
    class_lengths = {
        character: len(os.listdir(project_dir / f"data/raw/train/{character}"))
        for character in os.listdir(project_dir / "data/raw/train")
    }
    class_lengths = pd.DataFrame(class_lengths.items(), columns=["character", "images"]).sort_values(
        by="images", ascending=False
    )

    set_custom_template("custom_white")
    set_custom_show_config()
    get_top_classes_by_images(class_lengths, 10, "max")
    get_top_classes_by_images(class_lengths, 10, "min")


if __name__ == "__main__":
    main()
