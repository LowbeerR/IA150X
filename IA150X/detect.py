import os

import torch
import time
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import v2
from cnn import ConvNet, CustomImageDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
model.load_state_dict(torch.load('model.pth'))


classes = ('No hidden data', 'Hidden data')

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(256, 256)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == "__main__":
    while True:
        path = str(input("Paste the path\n"))
        if path == "test":
            batch_size = 32
            testset = CustomImageDataset(annotations_file='C:/Users/Rikard/Documents/GitHub/IA150X/IA150X/dataset2/static_dataset.csv', img_dir='C:/Users/Rikard/Documents/GitHub/IA150X/IA150X/dataset2/data3', transform=transforms)
            testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
            model.eval()
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                hidden = 0
                for images, labels in testset_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
                    hidden += (predicted == 1).sum().item()
                acc = 100.0 * n_correct / n_samples
                print(f'Accuracy of the network: {acc} %, Hidden data chance: {100*hidden/n_samples:.0f}%')
        if path == "test2":
            video_folder = input("Enter the evaluation dataset path \n")[1:-1]

            try:
                for videos in os.listdir(video_folder):
                    video_reader = torchvision.io.VideoReader(os.path.join(video_folder, videos), "video")
                    hidden = 0
                    n_samples = 0
                    print("checking video for hidden data")
                    for entry in video_reader:  # video[0]:
                        if n_samples >= 6000:
                            break
                        frame = entry['data']
                        frame = transforms(frame).unsqueeze(0).to(device)
                        model.eval()
                        with torch.no_grad():
                            outputs = model(frame)
                            _, predicted = torch.max(outputs, 1)
                            n_samples += 1
                            if predicted == 1:
                                hidden += 1
                    time2 = time.time()
                    print(
                        f"Hidden data chance: {100 * hidden / n_samples:.0f}%, Nr_frames: {n_samples}")
            except Exception as e:
                print("Error")
        else:
            try:
                image = read_image(path[1:-1])
            except Exception as e:
                # print(f"Error: {e}")
                try:
                    # video = read_video(path[1:-1],start_pts=0, end_pts=100, pts_unit='sec', output_format="TCHW")
                    time1 = time.time()
                    video_reader = torchvision.io.VideoReader(path[1:-1], "video")
                except Exception as f:
                    print(f"Error: {e, f}")
                else:
                    hidden = 0
                    n_samples = 0
                    print("checking video for hidden data")
                    for entry in video_reader: # video[0]:
                        if n_samples >= 6000:
                            break
                        frame = entry['data']
                        frame = transforms(frame).unsqueeze(0).to(device)
                        model.eval()
                        with torch.no_grad():
                            outputs = model(frame)
                            _, predicted = torch.max(outputs, 1)
                            n_samples += 1
                            if predicted == 1:
                                hidden += 1
                    time2 = time.time()
                    print(f"Hidden data chance: {100*hidden/n_samples:.0f}%, Time: {time2-time1:} Nr_frames: {n_samples}")
            else:
                image = transforms(image).unsqueeze(0).to(device)
                img = torchvision.utils.make_grid(image).to('cpu')
                plt.imshow(img.numpy().transpose((1, 2, 0)))
                plt.show()
                model.eval()
                with torch.no_grad():
                    output = model(image)

                _, predicted_class = torch.max(output, 1)
                print("Predicted Class:", classes[predicted_class.item()])
