import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from custom_plotly_templates import set_custom_show_config, set_custom_template
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.utils import get_logger, plotly_fig_to_array

set_custom_template("custom_white")
set_custom_show_config()

logger = get_logger()


inverse_transform = transforms.Normalize(
    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    [1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def add_sample_images(writer: SummaryWriter, dataset: ImageFolder) -> go.Figure:
    classes = np.array(dataset.classes)

    random_idxs = torch.randperm(len(dataset))[:8]
    images = torch.stack([dataset[i][0] for i in random_idxs])
    images = inverse_transform(images).permute(0, 2, 3, 1).numpy()
    images = (images * 255).astype("uint8")  # type: ignore
    labels = classes[torch.tensor(dataset.targets)[random_idxs]]

    fig = px.imshow(
        images,
        facet_col=0,
        facet_col_wrap=4,
        binary_string=True,
    )

    fig.update_layout(coloraxis_showscale=False, margin=dict(t=50), width=1000)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    for annotation, label in zip(fig.layout.annotations[:4], labels[4:8]):
        label = label.replace("_", " ").title()
        annotation.update(y=annotation.y + 0.0, text=label, font_size=16)

    for annotation, label in zip(fig.layout.annotations[4:8], labels[:4]):
        label = label.replace("_", " ").title()
        annotation.update(y=annotation.y + 0.0, text=label, font_size=16)

    fig_array = plotly_fig_to_array(fig)
    fig_array = fig_array.transpose((2, 0, 1))
    writer.add_image("Sample images", fig_array)
    writer.flush()
    logger.info("added sample images to tensorboard")

    return fig


def add_embeddings(writer: SummaryWriter, dataset: ImageFolder, n_images: int = 100) -> None:
    classes = np.array(dataset.classes)
    targets = np.array(dataset.targets)

    random_idxs = torch.randperm(len(dataset))[:n_images]
    images = torch.stack([dataset[i][0] for i in random_idxs])
    images = inverse_transform(images)
    labels = classes[targets[random_idxs]]

    writer.add_embedding(images.view(n_images, -1), metadata=labels, label_img=images, global_step=0)
    writer.flush()
    logger.info("added image embeddings to tensorboard")


def add_model(writer: SummaryWriter, model: torch.nn.Module, images: torch.Tensor) -> None:
    writer.add_graph(model, images)
    writer.flush()
    logger.info("added model graph to tensorboard")


def add_sample_predictions(
    writer: SummaryWriter,
    images: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    classes: np.ndarray,
    epoch: int,
) -> go.Figure:

    predictions = torch.argmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    probs = [prob[prediction] for prob, prediction in zip(probs, predictions)]  # type: ignore

    images = inverse_transform(images).permute(0, 2, 3, 1).cpu().numpy()
    images = (images * 255).astype("uint8")  # type: ignore

    labels = classes[labels.cpu()]
    predictions = classes[predictions.cpu()]

    fig = px.imshow(images, facet_col=0, facet_col_wrap=4, facet_col_spacing=0.05, width=1000)

    fig.update_layout(coloraxis_showscale=False, margin=dict(t=0, b=0))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    text_template = "{prediction}, {prob:.2%}<br>Label: {label}"

    for annotation, prediction, prob, label in zip(fig.layout.annotations, predictions, probs, labels):
        text = text_template.format(
            prediction=prediction,
            prob=prob,
            label=label,
        )

        annotation.update(y=0.82, text=text, font_color="green" if prediction == label else "red")

    fig_array = plotly_fig_to_array(fig)
    fig_array = fig_array.transpose((2, 0, 1))
    writer.add_image("Predictions vs. Actual", fig_array, epoch)
    writer.flush()

    return fig
