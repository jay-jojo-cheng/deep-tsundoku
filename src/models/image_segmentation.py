"""
This file contains functions that allow detecting book spines
in images of bookshelves using OpenCV. The code is highly inspired by
https://github.com/LakshyaKhatri/Bookshelf-Reader-API
"""

import math

import cv2
import numpy as np
from PIL import Image


def crop_book_spines_in_image(pil_img, output_img_type: str = "pil"):
    """
    Identifies the book spines in the input image
    and returns list of such book spine images.
    """
    cv2_img = pil_image_to_opencv_image(pil_img)
    # resizing images to control cropping behavior
    cv2_img = resize_img(cv2_img)
    points = detect_spines(cv2_img)
    return get_cropped_images(cv2_img, points, output_img_type=output_img_type)


def detect_spines(img):
    """
    Returns a list of lines separating
    the detected spines in the image
    """
    img = img.copy()
    height, width, _ = img.shape

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 70)

    # kernel = np.ones((4, 1), np.uint8)
    kernel = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    img_erosion = cv2.erode(edge, kernel, iterations=1)

    lines = cv2.HoughLines(img_erosion, 1, np.pi / 180, 100)
    if lines is None:
        return []

    points = get_points_in_x_and_y(lines, width, height)

    points = remove_diagonals(points)

    points = shorten_line(points, height)

    points.sort(key=lambda val: val[0][0])

    points = remove_duplicate_lines(points)

    return points


def get_cropped_images(image, points, output_img_type: str = "pil"):
    """
    Takes a spine line drawn image and
    returns a list of opencv images split
    from the drawn lines
    """
    if output_img_type not in ["pil", "opencv"]:
        raise ValueError(
            f"`output_img_type` {output_img_type} not supported! "
            f"Types supported: 'pil' and 'opencv'"
        )

    image = image.copy()
    y_max, _, _ = image.shape
    last_x1 = 0
    last_x2 = 0
    cropped_images = []

    for point in points:
        ((x1, y1), (x2, y2)) = point

        crop_points = np.array([[last_x1, y_max], [last_x2, 0], [x2, y2], [x1, y1]])

        # Crop the bounding rect
        rect = cv2.boundingRect(crop_points)
        x, y, w, h = rect
        cropped = image[y : y + h, x : x + w].copy()

        # make mask
        crop_points = crop_points - crop_points.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [crop_points], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        # rotations
        dst = cv2.rotate(dst, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        # dst = cv2.rotate(dst, cv2.ROTATE_180)

        if output_img_type == "pil":
            dst = opencv_image_to_pil_image(dst)
        cropped_images.append(dst)

        last_x1 = x1
        last_x2 = x2

    return cropped_images


def get_points_in_x_and_y(hough_lines, max_x, max_y):
    """
    Takes a list of trigonometric form of lines
    and returns their starting and ending
    co-ordinates
    """
    points = []
    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + (max_y + 100) * (-b))
        y1 = int(y0 + (max_y + 100) * (a))
        start = (x1, y1)

        x2 = int(x0 - (max_y + 100) * (-b))
        y2 = int(y0 - (max_y + 100) * (a))
        end = (x2, y2)

        points.append((start, end))

    # Add a line at the very end of the image
    points.append(((max_x, max_y), (max_x, 0)))

    return points


def remove_duplicate_lines(sorted_points):
    """
    Searches for the lines that are drawn
    over each other in the image and returns
    a list of non duplicated line coordinates
    """
    last_x1 = 0
    last_x2 = 0
    non_duplicate_points = []
    for point in sorted_points:
        ((x1, y1), (x2, y2)) = point
        if last_x1 == 0 and x1 > 0:
            non_duplicate_points.append(point)
            last_x1 = x1

        # Ignore lines that start too close to previous line
        # and lines that intersect with previous line
        elif abs(last_x1 - x1) >= 25 and x2 > last_x2:
            non_duplicate_points.append(point)
            last_x1 = x1
            last_x2 = x2

    return non_duplicate_points


def remove_diagonals(points):
    """
    Filters for the lines that are at an angle
    superior to approx. 70 degrees and returns
    a list containing line coordinates
    """
    non_diagonals = []
    for point in points:
        ((x1, y1), (x2, y2)) = point
        if x1 == x2:
            non_diagonals.append(point)
        # slope > tan(70)
        # tan(70) is approx. 2.7
        elif abs((y2 - y1) / (x2 - x1)) > 2.7:
            non_diagonals.append(point)

    return non_diagonals


def shorten_line(points, y_max):
    """
    Takes a list of starting and ending
    coordinates of different lines
    and returns their trimmed form matching
    the image height
    """
    shortened_points = []
    for point in points:
        ((x1, y1), (x2, y2)) = point

        # Slope
        try:
            m = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            m = -1  # Infinite slope

        if m == -1:
            shortened_points.append(((x1, y_max), (x1, 0)))
            continue

        # From equation of line:
        # y-y1 = m (x-x1)
        # x = (y-y1)/m + x1
        # let y = y_max
        new_x1 = math.ceil(((y_max - y1) / m) + x1)
        start_point = (abs(new_x1), y_max)

        # Now let y = 0
        new_x2 = math.ceil(((0 - y1) / m) + x1)
        end_point = (abs(new_x2), 0)

        shortened_points.append((start_point, end_point))

    return shortened_points


def resize_img(img):
    """
    Resizes image to a max width or height of 1000px
    """
    img = img.copy()
    img_ht, img_wd, _ = img.shape

    max_lenght = 1000

    if img_wd >= img_ht:
        ratio = img_wd / img_ht
        new_width = 1000
        new_height = math.ceil(new_width / ratio)

    elif img_wd < img_ht:
        ratio = img_ht / img_wd
        new_height = 1000
        new_width = math.ceil(new_height / ratio)

    resized_image = cv2.resize(img, (new_width, new_height))

    return resized_image


def pil_image_to_opencv_image(pil_image):
    """
    Converts image from PIL Image to Opencv Image
    """
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def opencv_image_to_pil_image(opencv_image):
    """
    Converts image from Opencv Image to PIL Image
    """
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(opencv_image)

    return pil_image
