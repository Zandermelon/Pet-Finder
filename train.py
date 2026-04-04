import torch
import torch.nn as nn
import timm
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses, miners

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PetDataset(Dataset):
    def __init__(self, positive_folders, negative_folders, transform):
        self.transform = transform
        self.images = []
        self.labels = []

        for folder in positive_folders:
            if not os.path.exists(folder):
                print(f"Warning: {folder} not found, skipping...")
                continue
            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.images.append(os.path.join(folder, filename))
                    self.labels.append(0)

        for folder in negative_folders:
            if not os.path.exists(folder):
                print(f"Warning: {folder} not found, skipping...")
                continue
            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.images.append(os.path.join(folder, filename))
                    self.labels.append(1)

        print(f"Dataset: {self.labels.count(0)} positive, {self.labels.count(1)} negative photos")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

class PetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.embedding = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

def train():
    print("Setting up training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = PetDataset(
        positive_folders=[
            "data/training/cats/same_cat_cropped",
            "data/training/dogs/same_dog_cropped",
        ],
        negative_folders=[
            "data/training/cats/different_cats_cropped",
            "data/training/dogs/different_dogs_cropped",
        ],
        transform=train_transform
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = PetEmbedder().to(device)
    loss_fn = losses.TripletMarginLoss(margin=0.2)
    miner = miners.MultiSimilarityMiner()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("Starting training...")
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batches = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            hard_pairs = miner(embeddings, labels)
            loss = loss_fn(embeddings, labels, hard_pairs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")

    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/pet_embedder.pth")
    print("Training done! Model saved!")