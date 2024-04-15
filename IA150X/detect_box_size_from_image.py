from csv import reader, DictReader
import math
import os
import shutil
import time
import cv2
import ffmpeg
from pymediainfo import MediaInfo


def check_box_same_color(size, array, x_pos, y_pos):
    if x_pos + size > len(array[0]) or y_pos + size > len(array):
        return False
    color = array[y_pos][x_pos]
    for y in range(y_pos, y_pos + size):
        for x in range(x_pos, x_pos + size):
            if array[y][x] != color:
                return False
    return True


def no_adjacent_pixels_same_color(size, array, x_pos, y_pos):
    if size == 0 or size >= len(array) or size >= len(array[0]):
        raise Exception("size must be larger than 0 or smaller than the list")

    # checks that the box is of same color
    if not (check_box_same_color(size, array, x_pos, y_pos)):
        return False

    color = array[y_pos][x_pos]
    # up
    for x in range(x_pos, x_pos + size):
        new_y_pos = y_pos - 1
        if 0 <= new_y_pos <= len(array):
            if array[new_y_pos][x] == color:
                return False
    # down
    for x in range(x_pos, x_pos + size):
        new_y_pos = y_pos + size
        if new_y_pos < len(array):
            if array[new_y_pos][x] == color:
                return False
    # left
    for y in range(y_pos, y_pos + size):
        new_x_pos = x_pos - 1
        if 0 <= new_x_pos <= len(array[0]):
            if array[y][new_x_pos] == color:
                return False
    # right
    for y in range(y_pos, y_pos + size):
        new_x_pos = x_pos + size
        if new_x_pos < len(array[0]):
            if array[y][new_x_pos] == color:
                return False
    return True


def find_box_size(imm_arr):
    pixels = len(imm_arr[0]) * len(imm_arr)
    size_limit = 6
    for size in range(1, size_limit + 1):
        box_found_count = 0
        threshold = (pixels / (size * size)) * 0.01
        for y in range(imm_arr.shape[0] - size + 1):
            for x in range(imm_arr.shape[1] - size + 1):
                if no_adjacent_pixels_same_color(size, imm_arr, y, x):
                    box_found_count += 1
                    if box_found_count >= threshold:
                        return True
    return False


def video_to_frames(video_data, frames_checked_count, contains_data_threshold_ratio):
    path, type, hidden_data = video_data
    if 0 >= contains_data_threshold_ratio > 1:
        raise Exception("contains_data_threshold_procent must be between 0 and 1")

    temp_dir = os.path.join(os.curdir, "temp")
    try:
        os.mkdir(temp_dir)
    except OSError:
        shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
    total_fps = get_total_nr_frames(path)
    freq = total_fps // frames_checked_count
    # +4 to avoid first frames that contains "instructions" for ISG
    frames_indices = [(4 + i * freq) % total_fps for i in range(frames_checked_count)]
    for frame_index in frames_indices:
        # % 30000 exists due to limitation in ffmpeg, finding a specific frame beyond 30 000 causes the program to crash
        output_file = os.path.join(temp_dir, f"frame_{frame_index % 30000}.png")
        (
            ffmpeg
            .input(path)
            .filter('select', f'eq(n,{frame_index % 30000})')  # .format(frame_index))
            .output(output_file, vframes=1, loglevel="quiet")
            .run()
        )
    prediction = check_multiple_frames(temp_dir, frames_checked_count, contains_data_threshold_ratio)
    if prediction != -1:
        if prediction == int(hidden_data):
            return True
        else:
            return False
    else:
        raise Exception("Error in check_multiple_frames")


def check_multiple_frames(folder_path, frames_checked_count, contains_data_threshold_ratio):
    frames_needed_to_contain_data = math.floor(frames_checked_count * contains_data_threshold_ratio)
    frame = 0
    has_box_count = 0
    for image in os.listdir(folder_path):
        # Sets limit on when a video is classified as hidden data
        image_path = os.path.join(folder_path, image)
        black_white = create_black_white_picture(image_path)
        frame += 1
        if find_box_size(black_white):
            has_box_count += 1
        if has_box_count >= frames_needed_to_contain_data:
            return 1
        if frame == frames_checked_count and has_box_count < frames_needed_to_contain_data:
            return 0
    return -1  # error


def get_total_nr_frames(file_name):
    fps = get_fps(file_name)
    if fps == 0:
        raise Exception(f"Fps of video is 0, error {file_name}")
    return get_fps(file_name) * get_length(file_name)


def get_length(file_name):
    duration = 0
    media_info = MediaInfo.parse(file_name)
    for track in media_info.tracks:
        if track.track_type == "Video":
            duration = track.duration
    if duration == 0:
        raise Exception(f"{file_name} is 0s long")
    return math.floor(duration / 1000)  # Convert milliseconds to seconds


def get_fps(file_name):
    fps = 0
    data = MediaInfo.parse(file_name)
    for track in data.tracks:
        if track.track_type == "Video":
            fps = float(track.frame_rate)
    return int(math.floor(fps))


def create_black_white_picture(image_path):
    im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    (_, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw


def test_videos_from_csv(file_csv, nr_of_frames_checked, data_ratio):
    time1 = time.time()
    with open(file_csv, "r", encoding="utf-8") as csvfile2:
        reader1 = reader(csvfile2)
        tot_videos = sum(1 for _ in reader1) - 1
    with open(file_csv, 'r', encoding="utf-8") as csvfile:
        path = "evaluation_dataset/"
        reader_csv = DictReader(csvfile)
        correct = 0
        correct_per_type = {'static_bw': {'no_hidden_data': 0, 'hidden_data': 0},
                            'static_rgb': {'no_hidden_data': 0, 'hidden_data': 0},
                            'other': {'no_hidden_data': 0, 'hidden_data': 0}}
        false = 0
        false_per_type = {'static_bw': {'no_hidden_data': 0, 'hidden_data': 0},
                          'static_rgb': {'no_hidden_data': 0, 'hidden_data': 0},
                          'other': {'no_hidden_data': 0, 'hidden_data': 0}}
        nr = 0

        for row in reader_csv:
            nr = nr + 1
            name = path + row['name']
            type = row['type'].replace(" ", "")
            hidden_data = int(row['hidden_data'])
            video_info = (name, type, hidden_data)
            print(f"\r{100 * nr // tot_videos}%  │{'█' * nr}{'-' * (tot_videos - nr)}│", end='')
            if video_to_frames(video_info, nr_of_frames_checked, data_ratio):
                correct = correct + 1
                if hidden_data == 1:
                    correct_per_type[type]['hidden_data'] = correct_per_type[type]['hidden_data'] + 1
                else:
                    correct_per_type[type]["no_hidden_data"] = correct_per_type[type]["no_hidden_data"] + 1
            else:
                false = false + 1
                if hidden_data == 1:
                    false_per_type[type]['hidden_data'] = false_per_type[type]['hidden_data'] + 1
                else:
                    false_per_type[type]['no_hidden_data'] = false_per_type[type]['no_hidden_data'] + 1
        print(f"\nNr of correct samples: {correct}, Number of false samples: {false}"
              f"\nNr of correct per type: static_bw = {correct_per_type['static_bw']}, static_rgb = {correct_per_type['static_rgb']}, other = {correct_per_type['other']}"
              f"\nNr of false per type: static_bw = {false_per_type['static_bw']}, static_rgb = {false_per_type['static_rgb']}, other = {false_per_type['other']}"
              f"\nTotal accuracy: {100 * correct / (correct + false):.0f}%"
              f"\nTime elapsed: {(time.time() - time1) / 60:.2f} minutes")


if __name__ == "__main__":
    # get list of coordinates that has boxes
    nr_of_frames_to_be_checked = 10
    contains_data_ratio = 0.9  # 10*0.9 = 9 frames needs to contain hidden_data to be classified as "hidden_data"

    test_videos_from_csv("eval.csv", nr_of_frames_to_be_checked, contains_data_ratio)
