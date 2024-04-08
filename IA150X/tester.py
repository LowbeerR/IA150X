import os.path

from pytube import YouTube
from csv import DictWriter

#appends title name to existing txt file
def add_names_to_csv(path):
    try:
        with open(path, "r") as file:
            lines = file.readlines()
        with open(path, "w", encoding="utf-8") as file:
            for line in lines:
                url = line.strip().split(',')[0]
                video_title = YouTube(url).title
                modif_line = line.strip() + ", " + video_title + "\n"
                file.write(modif_line)
    except Exception as e:
        print(e)


def download_videos(path):
    nr = 0
    try:
        with open(path, "r") as file:
            for line in file:
                nr = nr + 1
                URL = line.strip().split(',')[0]
                name = YouTube(URL).title + ".mp4"
                name = name.replace("|", "").replace("?", "").replace('"', "").replace("*", "")
                if not os.path.exists(os.path.join(out_folder, name)):
                    YouTube(URL).streams.first().download(output_path=out_folder, filename=name)
                data.append({'name': name, 'URL': URL, 'hidden_data': 0, 'type': line.strip().split(',')[1]})
                print(f"Downloaded {nr} videos")
    except FileNotFoundError as e:
        print(e)


out_folder = 'evaluation_dataset'

if __name__ == "__main__":
    while True:
        headers = None
        data = None
        path = str(input("Enter the path to the dataset\n"))
        try:
            headers = ['name', 'URL', 'hidden_data', 'type']
            data = []
            download_videos(path[1:-1])
        except Exception as e:
            print(e)
        if headers is not None and data is not None:
            with open('eval.csv', 'w', newline='', encoding="UTF-8") as csv:
                writer = DictWriter(csv, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)