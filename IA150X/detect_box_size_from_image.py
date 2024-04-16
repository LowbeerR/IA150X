from csv import reader, DictReader
import numpy as np
import math
import time
import av
from numba import jit


@jit
def check_box_same_color(size, array, x_pos, y_pos):
    if size == 0 or size >= len(array) or size >= len(array[0]):
        raise Exception("size must be larger than 0 or smaller than the list")
    color = array[y_pos][x_pos]

    for y in range(y_pos, y_pos + size):
        for x in range(x_pos, x_pos + size):
            if array[y][x] != color:
                return False
    return True


@jit
def no_adjacent_pixels_same_color(size, array, x_pos, y_pos):
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


@jit
def find_box_size(imm_arr):
    pixels = len(imm_arr[0]) * len(imm_arr)
    size_limit = 4
    for size in range(1, size_limit + 1):
        box_found_count = 0
        threshold = (pixels / (size * size)) * threshold_varible_tuning
        for y in range(imm_arr.shape[0] - size + 1):
            for x in range(imm_arr.shape[1] - size + 1):
                if no_adjacent_pixels_same_color(size, imm_arr, y, x):
                    box_found_count += 1
                    if box_found_count >= threshold:
                        return True
    return False


@jit
def top_left_crop(img, target_width, target_height):
    (w, h) = img.shape
    if w < target_width:
        target_width = w
    if h < target_height:
        target_height = h
    img = img[0:target_width, 0:target_height]
    return img


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
            if video_to_frames(video_info, nr_of_frames_checked,
                               data_ratio):  # video_to_frames(video_info, nr_of_frames_checked, data_ratio):
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


# https://stackoverflow.com/questions/51285593/converting-an-image-to-grayscale-using-numpy
def black_white_conversion(image):
    grayValue = 0.07 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.21 * image[:, :, 0]
    img = grayValue.astype(np.uint8)
    img[img > 128] = 255
    img[img <= 128] = 0
    return img


@jit
def top_left_crop_alt(img, target_width, target_height):
    (w, h, c) = img.shape
    if w < target_width:
        target_width = w
    if h < target_height:
        target_height = h
    img = img[0:target_width, 0:target_height]
    return img


def detect_hidden_data_video(path, frames_checked_count, contains_data_threshold_ratio):
    container = av.open(path)
    # container.streams.video[0].thread_type = "AUTO"  # Go faster!

    frames = 0
    index = 0
    has_box_count = 0
    total_frames = container.streams.video[0].frames
    if frames_checked_count > (total_frames - 4):
        frames_checked_count = total_frames - 5
    frames_needed_to_contain_data = math.floor(frames_checked_count * contains_data_threshold_ratio)

    # +4 to avoid first frames that contains "instructions" for ISG

    for frame in container.decode(video=0):
        # Sets limit on when a video is classified as hidden data

        if index > 4:
            frames += 1
            imm = frame.to_ndarray(format="rgb24")
            imm = top_left_crop_alt(imm, width_crop, height_crop)
            imm = black_white_conversion(imm)
            if find_box_size(imm):
                has_box_count += 1
            if has_box_count >= frames_needed_to_contain_data:
                return 1
            if frames == frames_checked_count and has_box_count < frames_needed_to_contain_data:
                return 0
        else:
            index += 1

    return -1  # error


def video_to_frames(video_data, frames_checked_count, contains_data_threshold_ratio):
    path, type, hidden_data = video_data
    if 0 >= contains_data_threshold_ratio > 1:
        raise Exception("contains_data_threshold_procent must be between 0 and 1")
    prediction = detect_hidden_data_video(path, frames_checked_count, contains_data_threshold_ratio)
    if prediction != -1:
        if prediction == int(hidden_data):
            return True
        else:
            return False
    else:
        raise Exception(f"Error in check_multiple_frames, prediction: {prediction}")


if __name__ == "__main__":
    # get list of coordinates that has boxes

    nr_of_frames_to_be_checked = 4
    contains_data_ratio = 0.8  # 10*0.9 = 9 frames needs to contain hidden_data to be classified as "hidden_data"
    height_crop = 32
    width_crop = 32
    threshold_varible_tuning = 0.03  # in find_box_size_numpy

    test_videos_from_csv("eval.csv", nr_of_frames_to_be_checked, contains_data_ratio)
