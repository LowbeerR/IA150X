import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.tv_tensors._dataset_wrapper import wrap_dataset_for_transforms_v2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 50
learning_rate = 0.001

# Resize pictures
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transforms2 = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load Custom Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        image = image.to(device=device)
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(device=device)

        return image, label


# Create dataset
dataset = CustomImageDataset(annotations_file='train.csv', img_dir='training_images', transform=transforms)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Add dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms2)
for i in range(len(train_dataset.targets)):
    train_dataset.targets[i] = 0
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Combine datasets
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset])

combined_loader = DataLoader(combined_dataset, batch_size=batch_size)


def imshow(img):
    img = img.to(device='cpu')
    img = img / 2 + 0.5
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()


# Get some random training images
dataiter = iter(dataset_loader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))


def test_loop(dataloader):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


"""
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 10
        return x
"""


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6195200, 2),
        )

    def forward(self, x):
        return self.model(x)

    # Create model, loss_fn and optimizer


model = ConvNet().to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    model.train()
    i = 0
    for epoch in range(num_epochs):
        for batch in combined_loader:
            i += 2
            inputs, labels = batch
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            pred_label = model(inputs)
            loss = loss_fn(pred_label, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'model.pth')
