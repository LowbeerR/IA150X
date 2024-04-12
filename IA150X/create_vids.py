import random
import imageio
import numpy as np

from PIL import Image

width = 1280
height = 700
frame_amount = 600


def bw_movie(video_filename):
    frames = []
    for i in range(frame_amount):
        image = Image.new("1", (width, height))
        pixels = image.load()

        for x in range(width):
            for y in range(height):
                pixels[x, y] = (random.randint(0, 1))
        image_rbg = image.convert("RGB")
        image = np.array(image_rbg)
        frames.append(image)
        if (i + 1) % 60 == 0:
            print(f"Frame {i + 1} of {frame_amount}")
    imageio.mimsave(video_filename, frames, fps=10, codec="libx264")


def rgb_movie(video_filename):
    frames = []
    for i in range(frame_amount):
        image = Image.new("RGB", (width, height))
        pixels = image.load()

        for x in range(width):
            for y in range(height):
                pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = np.array(image)
        frames.append(image)
        if (i + 1) % 60 == 0:
            print(f"Frame {i + 1} of {frame_amount}")
    imageio.mimsave(video_filename, frames, fps=10, codec="libx264")
