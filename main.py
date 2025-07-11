import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from albumentations import Compose, Resize, Normalize, RandomCrop, HorizontalFlip, Rotate, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# Define the neural network architecture
class CatDogNet(nn.Module):
    def __init__(self):
        super(CatDogNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Data augmentation and transformation
transform = Compose([
    Resize(64, 64),
    RandomCrop(64, 64),
    HorizontalFlip(),
    Rotate(limit=15),
    RandomBrightnessContrast(),
    Normalize(),
    ToTensorV2()
])

# Load dataset
dataset = datasets.ImageFolder('PetImages', transform=lambda img: transform(image=np.array(img))['image'])
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training function
def train_model(data_dir):
    model = CatDogNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    writer = SummaryWriter()

    for epoch in range(20):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                pred = outputs.argmax(dim=1)
                val_correct += pred.eq(targets).sum().item()
                val_total += targets.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy = 100. * correct / total
        val_accuracy = 100. * val_correct / val_total

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(f'Epoch {epoch}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.2f}%')

        scheduler.step()

    torch.save(model.state_dict(), 'cat_dog_model.pth')
    writer.close()

# Test function
def test_image(image_path):
    model = CatDogNet()
    model.load_state_dict(torch.load('cat_dog_model.pth'))
    model.eval()

    transform = Compose([
        Resize(64, 64),
        Normalize(),
        ToTensorV2()
    ])

    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = transform(image=image)["image"].unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1)
        prob = output[0][pred].item()

    return "Cat" if pred.item() == 0 else "Dog", prob

# Example usage
if __name__ == "__main__":
    #train_model("PetImages")
    prediction, confidence = test_image("PetImages/Cat/1.jpg")
    print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")