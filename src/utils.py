import os
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt


def load_image(image_path: str):
    input_image = image.imread(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../images/' + image_path)
    )
    return input_image


def print_rgb_images(rgbimage):
    red, green, blue = rgbimage[:, :, 0], rgbimage[:, :, 1], rgbimage[:, :, 2]
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 15))
    ax1.imshow(red, cmap='Reds')
    ax2.imshow(green, cmap='Greens')
    ax3.imshow(blue, cmap='Blues')
    plt.show()


def convert_to_gray(rgbimage):
    return np.dot(rgbimage[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == '__main__':
    load_image("csivi.jpg")
