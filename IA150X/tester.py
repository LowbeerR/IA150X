from pytube import YouTube

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
    try:
        with open(path, "r") as file:
            for line in file:
                url = line.strip().split(',')[0]
                print(f"downloading video{url}")
                YouTube(url).streams.first().download(output_path=out_folder)
    except FileNotFoundError as e:
        print(e)


out_folder = 'evaluation_dataset'

if __name__ == "__main__":
        path = str(input("Enter the path to the dataset\n"))
        try:
            #download_videos(path[1:-1])
            add_names_to_csv(path[1:-1])
        except Exception as e:
            print(e)
