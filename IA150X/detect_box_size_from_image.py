import csv
import math
import os
import shutil
import cv2
import ffmpeg
from pymediainfo import MediaInfo

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
    pixels = len(imm_arr[0]) * len(imm_arr)
    size_limit = 6
    for size in range(1, size_limit + 1):
        box_found_count = 0
        threshold = (pixels / (size * size)) * 0.01
        for y in range(imm_arr.shape[0] - size + 1):
            for x in range(imm_arr.shape[1] - size + 1):
                if no_adjacent_pixels_same_color(size, imm_arr, y, x):
                    box_found_count += 1
                    # print(f"One box of size {size} found at (x,y) = ({y}, {x})")
                    if box_found_count >= threshold:
                        # print( f"Frame contains hidden data, found evidence on {box_found_count} places of size {size}")
                        return True
    return False


def video_to_frames(video_path, frames_checked_count, contains_data_threshold_ratio):
    if 0 >= contains_data_threshold_ratio > 1:
        raise Exception("contains_data_threshold_procent must be between 0 and 1")

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
    # +1 to avoid first frame that contains "instructions"
    frames_indices = [1 + i * freq for i in range(frames_checked_count)]
    print(frames_indices)
    for frame_index in frames_indices:
        # % 30000 exists due to limitation in ffmpeg, finding a specific frame beyond 30 000 causes the program to crash
        output_file = os.path.join(temp_dir, f"frame_{frame_index % 30000}.png")
        print(output_file, frame_index)
        (
            ffmpeg
            .input(video_path)
            .filter('select', f'eq(n,{frame_index % 30000})')  # .format(frame_index))
            .output(output_file, vframes=1, loglevel="quiet")
            .run()
        )
    check_multiple_frames(temp_dir, frames_checked_count * contains_data_threshold_ratio)


def check_multiple_frames(folder_path, frames_needed_to_contain_data):
    frame = 0
    has_box_count = 0
    for image in os.listdir(folder_path):
        # Sets limit on when a video is classified as hidden data
        print(f"Checking frame {frame}/ {frames_needed_to_contain_data}")
        image_path = os.path.join(folder_path, image)
        black_white = create_black_white_picture(image_path)
        frame += 1
        print(f"{image}")
        if find_box_size(black_white):
            has_box_count += 1
        if has_box_count == math.floor(frames_needed_to_contain_data):
            print("Video contains hidden data")
        elif frame == math.floor(frames_needed_to_contain_data):
            print("Video contains NO hidden data")


def get_total_nr_frames(file_name):
    fps = get_fps(file_name)
    if fps == 0:
        raise Exception(f"Fps of video is 0, error {file_name}")
    print(f"fps: {get_fps(file_name)}")
    print(f"length: {get_length(file_name)}")
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
    cv2.imwrite(image_path, im_bw)
    return im_bw


def test_videos_from_csv(file_csv):
    with open(file_csv, 'r', encoding="utf-8") as csvfile:
        path = "evaluation_dataset/"
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = path + row['name']
            video_to_frames(name, 2, 1)


if __name__ == "__main__":
    # get list of coordinates that has boxes
    test_videos_from_csv("eval.csv")

    ''' try:
        for file in os.listdir("videos"):
            # 1 = 100%, 0.5 = 50%
            video_to_frames("videos/" + file, 10, 1)
    except Exception as e:
        print(e)
    '''
