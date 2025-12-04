# utils.py
import cv2
import numpy as np
from config import IMG_WIDTH, IMG_HEIGHT

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_split_line(binary_image_inv):
    horizontal_projection = np.sum(binary_image_inv, axis=1) / 255
    h = len(horizontal_projection)
    start_search = int(h * 0.2)
    end_search = int(h * 0.8)
    projection_middle = horizontal_projection[start_search:end_search]
    if projection_middle.size == 0: return int(h / 2)
    min_val = np.min(projection_middle)
    min_indices = np.where(projection_middle == min_val)[0]
    if min_indices.size > 0: min_idx_relative = min_indices[len(min_indices) // 2]
    else: min_idx_relative = (start_search + end_search) // 2
    return start_search + min_idx_relative

def sort_contours_left_to_right(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
        key=lambda b:b[1][0], reverse=False))
    return (contours, bounding_boxes)

def preprocess_char(roi):
    padding = 3 
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
        roi = roi.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        roi = roi.astype('float32') / 255.0
        return roi
    return None