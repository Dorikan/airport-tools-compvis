import os
import time
import copy
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T


class EfficientNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Use updated weights parameter instead of deprecated pretrained
        self.base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = self.base.features
        self.pool = self.base.avgpool
        self.embedding_layer = nn.Flatten()
        in_features = self.base.classifier[1].in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.embedding_layer(x)
        if return_embedding:
            return x
        out = self.fc(x)
        return out


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, save_path, num_epochs=25, scheduler=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    log_file = os.path.join(save_path, "training_log.csv")
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        epoch_stats = {}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_stats[f"{phase}_loss"] = epoch_loss
            epoch_stats[f"{phase}_acc"] = epoch_acc.item()

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(save_path, "best_model.pth"))
                print(">> Saved best model.")

        # Save log
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1,
                             epoch_stats.get("train_loss", 0), epoch_stats.get("train_acc", 0),
                             epoch_stats.get("valid_loss", 0), epoch_stats.get("valid_acc", 0)])

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{epoch + 1}.pth"))
            print(f">> Saved checkpoint at epoch {epoch + 1}")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing train/ and valid/")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models and logs")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    train_transforms = T.Compose([
        T.RandomRotation(40),
        T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=20),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.RandomResizedCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    valid_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        print(f"Error: Train or valid directory not found in {args.data_dir}")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True),
        "valid": DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    }
    dataset_sizes = {"train": len(train_dataset), "valid": len(valid_dataset)}

    # Model setup
    model = EfficientNetWithEmbeddings(num_classes=args.num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)

    # Train
    train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, args.save_dir, args.num_epochs)


if __name__ == "__main__":
    main()
