
import os
import glob
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF

# CPU training
device = torch.device("cpu")
torch.set_num_threads(4)

# Dataset paths
possible_paths = [
    r"D:\DeepShield\image\images",
    r"D:\DeepShield\image",
    r"D:\DeepShield\images"
]

DATASET_PATH = next(
    (p for p in possible_paths if os.path.isdir(p)),
    None
)

if DATASET_PATH is None:
    raise FileNotFoundError("FFHQ image folder not found.")

image_files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*.jpg"),
    recursive=True
)

image_files += glob.glob(
    os.path.join(DATASET_PATH, "**", "*.jpeg"),
    recursive=True
)

image_files = list(set(image_files))

if not image_files:
    raise RuntimeError("No JPG images found in dataset folder.")

random.shuffle(image_files)
image_files = image_files[:500]

print("Dataset folder:", DATASET_PATH)
print("Images selected:", len(image_files))


class FaceDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files) * 2

    def __getitem__(self, index):
        image_index = index // 2
        is_augmented = index % 2

        image = Image.open(
            self.files[image_index]
        ).convert("RGB").resize((64, 64))

        if is_augmented:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.7, 1.3)
            )
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.7, 1.3)
            )
            image = TF.hflip(image) if random.random() > 0.5 else image

        image = TF.to_tensor(image)
        label = torch.tensor(is_augmented, dtype=torch.long)

        return image, label


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


dataset = FaceDataset(image_files)
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

model = SimpleFaceModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\nStarting experimental training...")

for epoch in range(2):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/2 | "
        f"Loss: {total_loss / len(loader):.4f}"
    )

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    os.path.join("models", "deep_shield_model.pth")
)

print("\nTraining completed!")
print("Model saved: models/deep_shield_model.pth")
