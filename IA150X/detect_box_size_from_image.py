import os
import shutil

import cv2
import ffmpeg
import numpy as np

_1x1 = "frame.png"
_2x2 = "frame3419.png"
_3x3 = "frame3419_with_3x3.png"


def check_box_same_color(size, array, x_pos, y_pos):
    if x_pos + size > len(array[0]) or y_pos + size > len(array):
        return False
    color = array[y_pos][x_pos]
    for y in range(y_pos, y_pos + size):
        for x in range(x_pos, x_pos + size):
            if array[y][x] != color:
                return False
    return True


# 1x1 boxes
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
    threshold = 5
    size_limit = 10

    for size in range(1, size_limit + 1):
        box_found_count = 0
        for y in range(imm_arr.shape[0] - size + 1):
            for x in range(imm_arr.shape[1] - size + 1):
                if no_adjacent_pixels_same_color(size, imm_arr, y, x):
                    box_found_count += 1
                    print(f"One box of size {size} found at (x,y) = ({y}, {x})")
                    if box_found_count >= threshold:
                        print("Contains hidden data")
                        return
        if box_found_count == 0:
            print(f"No box of size 0, 1 ... {size} found in image (No hidden data)")


0


def video_to_frames(video_path, frames_checked_count):
    temp_dir = os.path.join(os.curdir, "temp")
    try:
        os.mkdir(temp_dir)
    except OSError:
        shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
    total_fps = get_total_nr_frames(video_path)
    freq = total_fps // frames_checked_count
    print(total_fps)
    print(freq)
    frames_indices = [i * freq for i in range(frames_checked_count)]
    print(frames_indices)
    for frame_index in frames_indices:
        output_file = os.path.join(temp_dir, f"frame_{frame_index}.png")
        print(output_file)
        (
            ffmpeg
            .input(video_path)
            .filter('select', 'gte(n,{})'.format(frame_index))
            .output(output_file, vframes=1, format='image2', vcodec='mjpeg')
            .run()
        )
    check_multiple_frames(temp_dir)


def check_multiple_frames(folder_path):
    print("folder path", folder_path)
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)
        black_white = create_black_white_picture(image_path)
        find_box_size(black_white)


def get_total_nr_frames(file_name):
    return get_fps(file_name) * get_length(file_name)


def get_length(file_name):
    data = ffmpeg.probe(file_name)
    return int(data["streams"][0]["duration"].split(".")[0])


def get_fps(file_name):
    data = ffmpeg.probe(file_name)
    return int(data["streams"][0]["r_frame_rate"].split("/")[0])


def create_black_white_picture(image_path):
    im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    (_, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(image_path, im_bw)
    return im_bw


if __name__ == "__main__":
    # get list of coordinates that has boxes
    video_to_frames("lineup_clip_1_clip_1.mp4", 10)
