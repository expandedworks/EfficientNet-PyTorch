import os
import json

from PIL import Image
import torch
from torchvision import transforms

from .model import EfficientNet


def load_resources():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    model.eval()
    tfms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../examples/simple/labels_map.txt")) as f:
        labels_map = {int(index): label for index, label in json.load(f).items()}
    labels_list = [label for _, label in sorted(labels_map.items())]
    return {"model": model, "tfms": tfms, "labels_list": labels_list}


def predict_single_image(resources, image_path):
    model = resources["model"]
    labels_list = resources["labels_list"]
    tfms = resources["tfms"]

    img = tfms(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)

    prob = list(torch.softmax(outputs, dim=1)[0].numpy())
    class_probabilities = dict(zip(labels_list, prob))

    return class_probabilities
