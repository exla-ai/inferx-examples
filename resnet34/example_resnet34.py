import os
import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inferx.models.resnet34 import resnet34

def create_cifar_loaders(num_train_samples=500, batch_size=32):
    """Create CIFAR-10 data loaders with a subset of training data."""
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),  # More efficient resizing
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Store classes information
    classes = train_dataset.classes
    
    # Create a subset of 500 images
    indices = torch.randperm(len(train_dataset))[:num_train_samples]  # Faster than numpy
    train_subset = Subset(train_dataset, indices)
    
    # Create validation set (20% of the subset)
    val_size = int(0.2 * num_train_samples)
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # Create data loaders optimized for GPU
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # More workers for Orin
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_loader, val_loader, classes

def main():
    # Initialize the model
    model = resnet34()
    print(f"Using implementation: {model.__class__.__name__}")
    
    # Create data loaders with larger batch size for GPU
    train_loader, val_loader, classes = create_cifar_loaders(
        num_train_samples=500,
        batch_size=128  # Much larger batch size for GPU
    )
    print(f"Training on {len(train_loader.dataset)} images")
    print(f"Validating on {len(val_loader.dataset)} images")
    
    # Train the model and get training history
    start_time = time.time()
    history = model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        learning_rate=0.001,
        num_classes=len(classes)  # Pass number of classes explicitly
    )
    end_time = time.time()
    
    # Print metrics
    training_time = end_time - start_time
    print("\nTraining completed!")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Time per image: {training_time/500:.3f} seconds")
    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")

if __name__ == "__main__":
    main() 