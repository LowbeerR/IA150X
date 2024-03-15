import cv2


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
    size = 1
    threshhold = 5
    box_found_count = 0

    while box_found_count <= threshhold:
        box_found_count = 0
        for y in range(0, imm_arr.shape[0]):
            for x in range(0, imm_arr.shape[1]):
                if no_adjacent_pixels_same_color(size, imm_arr, y, x):
                    box_found_count += 1
                    print(f"One box of size {size} found at (x,y) = ({y}, {x})")
        size += 1


def create_black_white_picture(image_path):
    im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    (_, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw


if __name__ == "__main__":
    # get list of coordinates that has boxes
    bw = create_black_white_picture(input("Enter image name \n"))
    find_box_size(bw)
