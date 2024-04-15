import os
import time
from csv import reader

import torch
import torchvision
from torchvision.transforms import v2

from cnn import ConvNet

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
    try:
        time1 = time.time()
        with open("eval.csv", "r", encoding="utf-8") as csv:
            reader1 = reader(csv)
            tot_videos = sum(1 for _ in reader1) - 1
        with open("eval.csv", "r", encoding="utf-8") as csv:
            reader = reader(csv)
            next(reader, None)
            correct = 0
            correct_per_type = {'static_bw': {'no_hidden_data': 0, 'hidden_data': 0},
                                'static_rgb': {'no_hidden_data': 0, 'hidden_data': 0},
                                'other': {'no_hidden_data': 0, 'hidden_data': 0}}
            false = 0
            false_per_type = {'static_bw': {'no_hidden_data': 0, 'hidden_data': 0},
                              'static_rgb': {'no_hidden_data': 0, 'hidden_data': 0},
                              'other': {'no_hidden_data': 0, 'hidden_data': 0}}
            nr = 0
            print("checking videos for hidden data")
            for videos in reader:
                name = videos[0]
                label = int(videos[2])
                type = videos[3].replace(" ", "")
                video_reader = torchvision.io.VideoReader(os.path.join("evaluation_dataset", name), "video")
                hidden = 0
                nr = nr + 1
                n_samples = 0
                print(f"\r{100 * nr // tot_videos}%  │{'█' * nr}{'-' * (tot_videos - nr)}│", end='')
                for entry in video_reader:
                    if n_samples >= 6000:
                        break
                    frame = entry['data']
                    frame = transforms(frame).unsqueeze(0).to(device)
                    model.eval()
                    with torch.no_grad():
                        outputs = model(frame)
                        _, predicted = torch.max(outputs, 1)
                        n_samples += 1
                        if predicted.item() == 1:
                            hidden = hidden + 1
                if hidden / n_samples < 1:
                    if label == 0:
                        correct = correct + 1
                        correct_per_type[type]['no_hidden_data'] = correct_per_type[type]['no_hidden_data'] + 1
                    if label == 1:
                        false = false + 1
                        false_per_type[type]['hidden_data'] = false_per_type[type]['hidden_data'] + 1
                else:
                    if label == 0:
                        false = false + 1
                        false_per_type[type]['no_hidden_data'] = false_per_type[type]['no_hidden_data'] + 1
                    if label == 1:
                        correct = correct + 1
                        correct_per_type[type]['hidden_data'] = correct_per_type[type]['hidden_data'] + 1
            print(f"\nNr of correct samples: {correct}, Number of false samples: {false}"
                  f"\nNr of correct per type: static_bw = {correct_per_type['static_bw']}, static_rgb = {correct_per_type['static_rgb']}, other = {correct_per_type['other']}"
                  f"\nNr of false per type: static_bw = {false_per_type['static_bw']}, static_rgb = {false_per_type['static_rgb']}, other = {false_per_type['other']}"
                  f"\nTotal accuracy: {100 * correct / (correct + false):.0f}%"
                  f"\nTime elapsed: {(time.time() - time1) / 60:.2f} minutes")
    except Exception as e:
        print(f"Error: {e}")
