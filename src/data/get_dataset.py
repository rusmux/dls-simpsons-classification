from torchvision import transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=[-10, 10]),
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, hue=0.1),
    ]
)


def get_dataset(path: str) -> ImageFolder:
    return ImageFolder(path, transform=transform)
