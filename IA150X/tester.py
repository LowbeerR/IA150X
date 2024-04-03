from pytube import YouTube


def download_videos(path):
    try:
        with open(path, "r") as file:
            for line in file:
                name = line.strip().split(',')[0]
                YouTube(name).streams.first().download(output_path=out_folder)
    except FileNotFoundError as e:
        print(e)


out_folder = 'evaluation_dataset'

if __name__ == "__main__":
    while True:
        path = str(input("Enter the path to the dataset\n"))
        try:
            download_videos(path[1:-1])
        except Exception as e:
            print(e)
