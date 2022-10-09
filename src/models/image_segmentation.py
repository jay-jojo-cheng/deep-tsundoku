"""
This file contains functions that allow detecting book spines
in images of bookshelves using OpenCV. The code is highly inspired by
https://github.com/LakshyaKhatri/Bookshelf-Reader-API
"""

import math

import cv2
import numpy as np


def crop_book_spines_in_image(pil_img):
    """
    Identifies the book spines in the input image
    and returns list of such book spine images.
    """
    cv2_img = pil_image_to_opencv_image(pil_img)
    points = detect_spines(cv2_img)
    return get_cropped_images(cv2_img, points)


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
    points = get_points_in_x_and_y(lines, height)
    points.sort(key=lambda val: val[0][0])
    non_duplicate_points = remove_duplicate_lines(points)

    final_points = shorten_line(non_duplicate_points, height)

    return final_points


def get_cropped_images(image, points):
    """
    Takes a spine line drawn image and
    returns a list of opencv images split
    from the drawn lines
    """
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
        cropped_images.append(dst)

        last_x1 = x1
        last_x2 = x2

    return cropped_images


def get_points_in_x_and_y(hough_lines, max_y):
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
    points.append(((500, max_y), (500, 0)))

    return points


def remove_duplicate_lines(sorted_points):
    """
    Searches for the lines that are drawn
    over each other in the image and returns
    a list of non duplicated line coordinates
    """
    last_x1 = 0
    non_duplicate_points = []
    for point in sorted_points:
        ((x1, y1), _) = point
        if last_x1 == 0:
            non_duplicate_points.append(point)
            last_x1 = x1

        elif abs(last_x1 - x1) >= 25:
            non_duplicate_points.append(point)
            last_x1 = x1

    return non_duplicate_points


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


def pil_image_to_opencv_image(pil_image):
    """
    Converts image from PIL Image to Opencv Image
    """
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img
