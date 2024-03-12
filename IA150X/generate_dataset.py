import os
import ffmpeg
import csv
import shutil
import sys

path = "vid.avi"
path_to_videos = "videos"
target_dir = os.path.join(os.getcwd(), "data")
frames = sys.get_int_max_str_digits()


def count_files(rel_path):
    count = 0
    for _ in os.listdir(rel_path):
        count += 1
    return count


def generate_rows():
    rows = []
    for file in os.listdir(target_dir):
        rows.append({"file_path": file, "label": "1"})
    return rows


def generate_csv():
    with open("static_dataset.csv", "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file_path", "label"])
        rows = generate_rows()
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# rel path is location where frames are saved
def generate_frames(save_location_path, file_name):
    try:
        os.mkdir(save_location_path)
    except OSError:
        print("error")

    count = count_files(save_location_path)

    if count <= 0:
        ffmpeg.input(file_name).output(save_location_path + '\\frame%d.png', vframes=frames).run()
    else:
        try:
            os.mkdir("data_temp")
        except OSError:
            print("error")

        temp_dir = os.path.join(os.getcwd(), "data_temp")
        ffmpeg.input(file_name).output(temp_dir + '\\frame%d.png', vframes=frames).run()
        for file in os.listdir(temp_dir):
            shutil.move(temp_dir + "\\" + file, target_dir + "\\frame%d.png" % count)
            count += 1
        os.rmdir(temp_dir)


def zip_dataset(save_location_path):
    move_location = os.path.join(os.getcwd(), "dataset")
    try:
        os.mkdir(move_location)
    except OSError:
        print("")
    generate_csv()
    shutil.move(os.path.join(os.getcwd(), save_location_path), move_location)
    shutil.move(os.path.join(os.getcwd(), "static_dataset.csv"), move_location)
    shutil.make_archive("dataset", "zip", os.path.join(os.getcwd(), "dataset"))


def generate_frames_multiple_videos(save_location_path, path_to_video_folder):
    for file in os.listdir(path_to_video_folder):
        generate_frames(save_location_path, os.path.join(path_to_videos, file))

    if input("zip dataset & generate CSV? (yes/no)").lower() == "yes":
        try:
            zip_dataset(save_location_path)
            print("dataset.zip created!")
        except OSError:
            print("zipped dataset already exists")


if __name__ == "__main__":

    generate_frames_multiple_videos(input("save location path (./data): "), input("\nRelative path to videos (./videos): "))

