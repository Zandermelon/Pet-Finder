import torch
import torch.nn as nn
import timm
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses, miners

# The transform — same as before
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Augmentation transform for training
# This artificially creates more variety in your training data
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),           # random crop instead of center
    transforms.RandomHorizontalFlip(),    # randomly flip left/right
    transforms.ColorJitter(              # randomly adjust brightness/contrast
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class PetDataset(Dataset):
    def __init__(self, fiona_folder, other_cats_folder, transform):
        self.transform = transform
        self.images = []
        self.labels = []

        # Load Fiona's photos — label 0 means this is Fiona
        for filename in os.listdir(fiona_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(fiona_folder, filename)
                self.images.append(path)
                self.labels.append(0)  # 0 = Fiona

        # Load other cats — label 1 means not Fiona
        for filename in os.listdir(other_cats_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(other_cats_folder, filename)
                self.images.append(path)
                self.labels.append(1)  # 1 = not Fiona

        print(f"Dataset: {self.labels.count(0)} Fiona photos, {self.labels.count(1)} other cat photos")

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
        # Load EfficientNet as the base
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        
        # Add our own small embedding layer on top
        self.embedding = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # final fingerprint is 128 numbers
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)
        # Normalize so all fingerprints are on the same scale
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


def train():
    print("Setting up training...")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load dataset
    dataset = PetDataset(
        fiona_folder="data/cropped_fiona",
        other_cats_folder="data/cropped_other_cats",
        transform=train_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True  # shuffle so Fiona and other cats are mixed each batch
    )

    # Create the model
    model = PetEmbedder().to(device)

    # Triplet margin loss — this teaches the model to push 
    # different cats apart and pull same cats together
    loss_fn = losses.TripletMarginLoss(margin=0.2)
    
    # Miner finds the hardest examples to learn from in each batch
    miner = miners.MultiSimilarityMiner()

    # Optimizer — updates the model weights during training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train for 20 epochs
    print("Starting training...")
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batches = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Get embeddings for this batch
            embeddings = model(images)

            # Find hard examples using the miner
            hard_pairs = miner(embeddings, labels)

            # Calculate the loss
            loss = loss_fn(embeddings, labels, hard_pairs)

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")

    # Save the trained model
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/pet_embedder.pth")
    print("Training done! Model saved to data/models/pet_embedder.pth")