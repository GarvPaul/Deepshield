
import os
import torch
import torch.nn as nn


class SimpleFaceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)


def load_trained_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "deep_shield_model.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found: " + model_path
        )

    model = SimpleFaceModel()

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    model.eval()
    return model


def fgsm_attack(image, epsilon, data_gradient):
    perturbed_image = (
        image + epsilon * data_gradient.sign()
    )

    return torch.clamp(perturbed_image, 0, 1)


def generate_protected_image(image, epsilon=0.01):
    """
    Experimental FGSM perturbation using trained
    original-versus-augmented image classifier.
    Input shape: [1, 3, 64, 64]
    """

    model = load_trained_model()

    image = image.clone().detach()
    image.requires_grad = True

    output = model(image)

    target = torch.zeros(
        output.shape[0],
        dtype=torch.long
    )

    loss = nn.CrossEntropyLoss()(output, target)

    model.zero_grad()
    loss.backward()

    data_gradient = image.grad

    result = fgsm_attack(
        image,
        epsilon,
        data_gradient
    )

    return result.detach()
