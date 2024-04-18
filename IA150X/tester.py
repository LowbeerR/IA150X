import os.path
import shutil
from csv import DictWriter

from pytube import YouTube

from create_vids import bw_movie, rgb_movie
from docker import run_isg

out_folder = 'evaluation_dataset'


def download_videos(pth):
    nr = 0
    try:
        with open(pth, "r") as file:
            for line in file:
                nr = nr + 1
                URL = line.strip().split(',')[0]
                name = YouTube(URL).title + ".mp4"
                name = name.replace("|", "").replace("?", "").replace('"', "").replace("*", "").replace(",", "")
                if not os.path.exists(os.path.join(out_folder, name)):
                    YouTube(URL).streams.first().download(output_path=out_folder, filename=name)
                data.append({'name': name, 'URL': URL, 'hidden_data': 0, 'type': line.strip().split(',')[1]})
                print(f"Downloaded {nr} videos")
    except FileNotFoundError as e:
        print(e)


def make_isg():
    data = []
    shutil.copytree('evaluation_dataset', '..\\..\\Infinite-Storage-Glitch\\evaluation_dataset', dirs_exist_ok=True)
    with open('eval.csv', 'r', encoding="utf-8") as file:
        next(file, None)
        incr = 0
        if os.path.exists(f"evaluation_dataset/output.avi"):
            os.remove(f"evaluation_dataset/output.avi")
        for line in file:
            real_path = f"{out_folder}/{line.strip().split(',')[0]}"
            print(real_path)
            run_isg(incr % 3, real_path)
            shutil.move('..\\..\\Infinite-Storage-Glitch\\output.avi', 'evaluation_dataset')
            if os.path.exists(f"evaluation_dataset/output_{incr}.avi"):
                os.remove(f"evaluation_dataset/output_{incr}.avi")
            os.rename('evaluation_dataset/output.avi', f'evaluation_dataset/output_{incr}.avi')
            if incr % 3 == 2:
                type = "static_rgb"
            else:
                type = "static_bw"
            data.append({'name': f"output_{incr}.avi", 'URL': "NONE", 'hidden_data': 1, 'type': f"{type}"})
            incr = incr + 1
    with open('eval.csv', 'a', encoding="utf-8") as csv:
        writer = DictWriter(csv, fieldnames=headers, skipinitialspace=True, lineterminator='\n')
        writer.writerows(data)


if __name__ == "__main__":
    headers = None
    data = None
    path = 'yt_urls.txt'
    try:
        headers = ['name', 'URL', 'hidden_data', 'type']
        data = []
        download_videos(path)
        for i in range(0, 9):
            if not os.path.exists(os.path.join(out_folder, f"video_{i}.mp4")):
                bw = bw_movie(os.path.join(out_folder, f"video_{i}.mp4"), seed=i)
            data.append({'name': f"video_{i}.mp4", 'URL': "NONE", 'hidden_data': 0, 'type': "static_bw"})
            print(f"Created video {i + 1}")
        for i in range(0, 11):
            if not os.path.exists(os.path.join(out_folder, f"rgb_video_{i}.mp4")):
                bw = rgb_movie(os.path.join(out_folder, f"rgb_video_{i}.mp4"), seed=i)
            data.append({'name': f"rgb_video_{i}.mp4", 'URL': "NONE", 'hidden_data': 0, 'type': "static_rgb"})
            print(f"Created video {6 + i}")
    except Exception as e:
        print(e)
    if headers is not None and data is not None:
        with open('eval.csv', 'w', newline='', encoding="UTF-8") as csv:
            writer = DictWriter(csv, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)

        make_isg()
