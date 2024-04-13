import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import v2
from docker import run_isg
from generate_dataset import generate_frames_multiple_videos
import zipfile

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 50
learning_rate = 0.001

# Resize pictures
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(256, 256), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transforms2 = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(256, 256), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


def add_datasets():
    # Create dataset with ISG pictures
    dataset = CustomImageDataset(annotations_file='dataset/static_dataset.csv', img_dir='dataset/data2',
                                 transform=transforms)

    # Add CIFAR 10
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms2)
    for i in range(len(train_dataset.targets)):
        train_dataset.targets[i] = 0
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset])
    # Make loader for Model Training
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size)

    return dataset, train_dataset, combined_loader


def create_training_dataset():
    folder_path = '..\\..\\Infinite-Storage-Glitch\\src\\tests'
    zip_path = '..\\..\\Infinite-Storage-Glitch\\src\\tests.zip'
    output_path = '..\\..\\Infinite-Storage-Glitch\\output.avi'
    final_path = 'videos'
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as toZip:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    abspath = os.path.join(root, file)
                    relpath = os.path.relpath(abspath, folder_path)
                    toZip.write(abspath, relpath)
        run_isg(0, 'src/tests.zip')
        if os.path.exists(final_path):
            for root, _, files in os.walk(final_path):
                for file in files:
                    os.remove(os.path.join(root, file))
            os.removedirs(final_path)
        os.mkdir(final_path)
        shutil.move(output_path, final_path)
        os.rename(os.path.join(final_path, 'output.avi'), os.path.join(final_path, 'output_1.avi'))
        run_isg(1, 'src/tests.zip')
        shutil.move(output_path, final_path)
        generate_frames_multiple_videos('data2', 'videos')
        return add_datasets()
    except Exception:
        return None


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1016064, 2),
        )

    def forward(self, x):
        return self.model(x)

    # Create model, loss_fn and optimizer


model = ConvNet().to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    try:
        dataset, train_dataset, combined_loader = create_training_dataset()
    except Exception:
        print(f"Error: Make sure docker desktop is running and that it is installed in:"
              f"\n C:/Program Files/Docker/Docker/resources/bin/docker.exe also make sure that ISG is located in:"
              f"\n C:/Users/{os.getlogin()}/Documents/GitHub/Infinite-Storage-Glitch")
        exit(1)
    model.train()
    tot_len = (len(dataset) + len(train_dataset))
    for epoch in range(num_epochs):
        i = 0
        for batch in combined_loader:
            i += len(batch[0])
            inputs, labels = batch
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            pred_label = model(inputs)
            loss = loss_fn(pred_label, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 2000 == 0:
                print(f"\rLoading: {100 * i / tot_len:.0f} %", end='')
        print(f"\nEpoch: {epoch + 1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'model.pth')
