import os
import time
from csv import reader, DictReader

import torch
import torchvision
from torchvision.transforms import v2

from cnn import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
model.load_state_dict(torch.load('model.pth'))
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
classes = ('No hidden data', 'Hidden data')

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(144, 144)),
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
            reader = DictReader(csv)
            correct = 0
            false_positive = 0
            false_negative = 0
            total_correct_per_frame = 0
            total_frames = 0

            static_bw_video_correct = []
            static_rgb_video_correct = []
            other_video_correct = []
            static_bw_video_wrong = []
            static_rgb_video_wrong = []
            other_video_wrong = []
            correct_per_type = {'static_bw': {'no_hidden_data': 0, 'hidden_data': 0},
                                'static_rgb': {'no_hidden_data': 0, 'hidden_data': 0},
                                'other': {'no_hidden_data': 0, 'hidden_data': 0}}
            false = 0
            false_per_type = {'static_bw': {'no_hidden_data': 0, 'hidden_data': 0},
                              'static_rgb': {'no_hidden_data': 0, 'hidden_data': 0},
                              'other': {'no_hidden_data': 0, 'hidden_data': 0}}
            nr = 0
            other = 0
            bw = 0
            rgb = 0
            print("checking videos for hidden data")
            for row in reader:
                name = row['name']
                label = int(row['hidden_data'])
                type = row['type'].replace(" ", "")
                video_reader = torchvision.io.VideoReader(os.path.join("evaluation_dataset", name), "video")
                hidden = 0
                local_correct = 0
                local_incorrect = 0
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
                            if label == 1:
                                total_correct_per_frame += 1
                                local_correct += 1
                            else:
                                false_positive += 1
                                local_incorrect += 1
                        elif predicted.item() == 0:
                            if label == 0:
                                total_correct_per_frame += 1
                                local_correct += 1
                            else:
                                false_negative += 1
                                local_incorrect += 1
                    total_frames += 1
                if hidden / n_samples < 0.9:
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
                if type == 'other':
                    other_video_correct.append(local_correct)
                    other_video_wrong.append(local_incorrect)
                    other += 1
                elif type == 'static_bw':
                    static_bw_video_correct.append(local_correct)
                    static_bw_video_wrong.append(local_incorrect)
                    bw += 1
                elif type == 'static_rgb':
                    static_rgb_video_correct.append(local_correct)
                    static_rgb_video_wrong.append(local_incorrect)
                    rgb += 1

            print(f"\nNr of correct samples: {correct}, Number of false samples: {false}"
                  f"\nTotal correct frames: {total_correct_per_frame}, Number of false positives: {false_positive}, Number of false negatives: {false_negative}, Total amount of frames: {total_frames}"
                  f"\nNr of correct per type: static_bw = {correct_per_type['static_bw']}, static_rgb = {correct_per_type['static_rgb']}, other = {correct_per_type['other']}"
                  f"\nNr of false per type: static_bw = {false_per_type['static_bw']}, static_rgb = {false_per_type['static_rgb']}, other = {false_per_type['other']}"
                  f"\nTotal accuracy: {100 * correct / (correct + false):.0f}%"
                  f"\nTime elapsed: {(time.time() - time1) / 60:.2f} minutes"
                  f"\nOther correct per video: {other_video_correct}"
                  f"\nOther wrong per video: {other_video_wrong}"
                  f"\nBW correct per video: {static_bw_video_correct}"
                  f"\nBW wrong per video: {static_bw_video_wrong}"
                  f"\nRGB correct per video: {static_rgb_video_correct}"
                  f"\nRGB wrong per video: {static_rgb_video_wrong}"
                  )

    except Exception as e:
        print(f"Error: {e}")
