import os
import ffmpeg
import csv
import shutil
import sys
import json

path = "vid.avi"
#path_to_videos = "videos"
target_dir = os.path.join(os.getcwd(), "data2")
frames = sys.get_int_max_str_digits()


def count_files(rel_path):
    count = 0
    for _ in os.listdir(rel_path):
        count += 1
    return count


def generate_rows():
    rows = []
    for file in os.listdir(target_dir):
        if "rgb_" in file:
            rows.append({"file_path": file, "label": "2"})
        else:
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
        pass
        # print("error")

    count = count_files(save_location_path)

    if count <= 0:
        if "rbg_" in file_name:
            ffmpeg.input(file_name).output(save_location_path + '\\rgb_frame%d.png', vframes=frames, loglevel="quiet").run()
        else:
            ffmpeg.input(file_name).output(save_location_path + '\\frame%d.png', vframes=frames, loglevel="quiet").run()
    else:
        try:
            os.mkdir("data_temp")
        except OSError:
            pass
            #print("error")

        temp_dir = os.path.join(os.getcwd(), "data_temp")
        if "rgb_" in file_name:
            ffmpeg.input(file_name).output(temp_dir + '\\rgb_frame%d.png', vframes=frames, loglevel="quiet").run()
            for file in os.listdir(temp_dir):
                shutil.move(temp_dir + "\\" + file, target_dir + "\\rgb_frame%d.png" % count)
                count += 1
        else:
            ffmpeg.input(file_name).output(temp_dir + '\\frame%d.png', vframes=frames, loglevel="quiet").run()
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


def generate_frames_multiple_videos(save_location_path, path_to_video_folder):
    for file in os.listdir(path_to_video_folder):
        generate_frames(save_location_path, os.path.join(path_to_video_folder, file))

    try:
        zip_dataset(save_location_path)
        print("dataset created!")
    except OSError:
        shutil.rmtree('./dataset')
        zip_dataset(save_location_path)


def get_fps(file_name):
    data = ffmpeg.probe(file_name)
    return int(data["streams"][0]["r_frame_rate"].split("/")[0])


def get_total_nr_frames(file_name):
    return get_fps(file_name) * get_length(file_name)


def get_length(file_name):
    data = ffmpeg.probe(file_name)
    return int(data["streams"][0]["duration"].split(".")[0])


if __name__ == "__main__":
    generate_frames_multiple_videos("data2", "videos")
