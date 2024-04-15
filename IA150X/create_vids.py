import imageio
import numpy as np

width = 1280
height = 720
frame_amount = 600


def bw_movie(video_filename):
    frames = []
    for i in range(frame_amount):
        image = np.random.randint(0, 2, size=(height, width, 3), dtype=np.uint8) * 255
        frames.append(image)
        if (i + 1) % 60 == 0:
            print(f"Frame {i + 1} of {frame_amount}")
    imageio.mimsave(video_filename, frames, fps=10, codec="libx264")


def rgb_movie(video_filename):
    frames = []
    for i in range(frame_amount):
        image = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8) * 255
        frames.append(image)
        if (i + 1) % 60 == 0:
            print(f"Frame {i + 1} of {frame_amount}")
    imageio.mimsave(video_filename, frames, fps=10, codec="libx264")
