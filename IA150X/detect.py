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
from torchvision.transforms.v2 import ToPILImage

from cnn import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

classes = ('No hidden data', 'Hidden data')

transforms = v2.Compose([
    v2.CenterCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

if __name__ == "__main__":
    while True:
        path = str(input("Paste the path\n"))[1:-1]
        try:
            image = read_image(path)
        except Exception as e:
            print(f"Error: {e}")
        else:
            image = transforms(image).unsqueeze(0).to(device)
            img = torchvision.utils.make_grid(image).to('cpu')
            plt.imshow(img.numpy().transpose((1, 2, 0)))
            plt.show()
            with torch.no_grad():
                output = model(image)

            _, predicted_class = torch.max(output, 1)
            print("Predicted Class:", classes[predicted_class.item()])


