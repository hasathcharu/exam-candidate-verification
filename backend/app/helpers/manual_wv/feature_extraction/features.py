import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import radon
from math import atan2, pi


def get_num_of_black_pixels(binary_image):
    return np.sum(binary_image == 0)


def get_interior_contours(binary_image_inv):
    contours, hierarchy = cv2.findContours(
        binary_image_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    interior_areas = []
    num_interior_contours = 0
    interior_contours = np.zeros_like(binary_image_inv)

    if hierarchy is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] != -1:
                num_interior_contours += 1
                interior_areas.append(cv2.contourArea(contours[i]))
                cv2.drawContours(interior_contours, contours, i, (255, 255, 255), 3)
    mean = np.mean(interior_areas) if interior_areas else 0
    std = np.std(interior_areas) if interior_areas else 0
    return interior_contours, num_interior_contours, mean, std


def get_exterior_curves(binary_image_inv):
    contours, hierarchy = cv2.findContours(
        binary_image_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    num_exterior_curves = 0
    exterior_curves = np.zeros_like(binary_image_inv)
    if hierarchy is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] == -1:
                num_exterior_curves += 1
                cv2.drawContours(exterior_curves, contours, i, (255, 255, 255), 1)
    return exterior_curves, num_exterior_curves


def compute_chaincode_histogram(binary_image_inv):
    binary_image_inv = cv2.ximgproc.thinning(binary_image_inv)
    chaincode_map = {
        (1, 0): 0,
        (1, 1): 1,
        (0, 1): 2,
        (-1, 1): 3,
        (-1, 0): 4,
        (-1, -1): 5,
        (0, -1): 6,
        (1, -1): 7,
    }
    chaincode_histogram = np.zeros(8, dtype=int)

    contours, _ = cv2.findContours(
        binary_image_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    chaincode_images = [np.zeros_like(binary_image_inv) for _ in range(8)]
    color_image = np.ones_like(cv2.cvtColor(binary_image_inv, cv2.COLOR_GRAY2BGR)) * 255
    color_array = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    for contour in contours:
        for i in range(len(contour) - 1):
            x1, y1 = contour[i][0]
            x2, y2 = contour[i + 1][0]

            dx, dy = x2 - x1, y2 - y1
            chaincode_dir = chaincode_map.get((dx, dy), None)

            if chaincode_dir is not None:
                chaincode_histogram[chaincode_dir] += 1
                cv2.line(chaincode_images[chaincode_dir], (x1, y1), (x2, y2), 255, 1)
                cv2.line(color_image, (x1, y1), (x2, y2), color_array[chaincode_dir], 1)
    return chaincode_histogram, chaincode_images, color_image


def compute_stroke_width_histogram(binary_image_inv):
    skeleton = cv2.ximgproc.thinning(binary_image_inv)
    distance_transform = cv2.distanceTransform(binary_image_inv, cv2.DIST_L2, 5)
    stroke_widths = distance_transform[skeleton > 0]
    hist, _ = np.histogram(stroke_widths, bins=6, range=(0, np.max(stroke_widths)))
    return hist, stroke_widths

def get_contour_areas(binary_inv):
    largest_interior_contour = 0
    largest_exterior_curve = 0
    contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(contours[i])
                if area > largest_interior_contour:
                    largest_interior_contour = area
    
    if hierarchy is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] == -1:
                area = cv2.contourArea(contours[i])
                if area > largest_exterior_curve:
                    largest_exterior_curve = area
    return largest_exterior_curve, largest_interior_contour

def compute_hod(binary_image, pixels_per_cell=(16, 16), orientations=9):
    fd, _ = hog(
        binary_image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(1, 1),
        visualize=True,
        feature_vector=False,
    )
    hod_histogram = np.sum(np.abs(np.diff(fd, axis=0)), axis=(0, 1))
    return np.squeeze(hod_histogram)

def compute_hog(binary_image, pixels_per_cell=(16, 16), orientations=9):
    fd, _ = hog(
        binary_image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(1, 1),
        visualize=True,
        feature_vector=False,
    )
    hog_histogram = np.sum(fd, axis=(0, 1))
    return np.squeeze(hog_histogram)

def contour_direction_pdf(binary_line_inv, step=5, bins=12):
    contours, _ = cv2.findContours(binary_line_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hist = np.zeros(bins, np.float32)
    for cnt in contours:
        pts = cnt[:,0,:]              # shape (N,2)
        for k in range(len(pts)-step):
            dx, dy = pts[k+step]-pts[k]
            θ = abs(atan2(dy, dx))    # 0–π
            b = min(int(θ / pi * bins), bins - 1)
            hist[b] += 1
    return hist / hist.sum()

def contour_hinge_pdf(binary_line_inv, step=5, n=12):
    contours, _ = cv2.findContours(binary_line_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hist = np.zeros((2*n,2*n), np.float32)
    for cnt in contours:
        pts = cnt[:,0,:]
        for k in range(step, len(pts)-step):
            dx1, dy1 = pts[k+step]-pts[k]
            dx2, dy2 = pts[k-step]-pts[k]
            θ1 = atan2(dy1, dx1) % (2*pi)
            θ2 = atan2(dy2, dx2) % (2*pi)
            if θ2 < θ1: θ1, θ2 = θ2, θ1
            i = int(θ1 / (2*pi) * 2*n)
            j = int(θ2 / (2*pi) * 2*n)
            hist[i,j] += 1
    # keep upper triangle, flatten:
    tri = hist[np.triu_indices(2*n)]
    return tri / tri.sum()

def run_length_pdf(bw, axis=0, L=60):
    # axis=0 rows (h), 1 columns (v)
    img = bw if axis==0 else bw.T
    hist = np.zeros(L, np.float32)
    for line in img:
        run = 0
        for p in line:
            if p==0: run+=1
            elif run:
                if run<=L: hist[run-1]+=1
                run=0
        if run and run<=L: hist[run-1]+=1
    return hist / hist.sum()

def compute_slant_angle_histogram(binary_inv, num_bins=9):
    sobelx = cv2.Sobel(binary_inv, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary_inv, cv2.CV_64F, 0, 1, ksize=3)
    sobelx[sobelx == 0] = 1e-6

    angles = np.rad2deg(np.arctan2(sobely, sobelx))
    angles = angles[(binary_inv > 0)]
    angles = np.clip(angles, -90, 90)
    hist, _ = np.histogram(angles, bins=num_bins, range=(-90, 90), density=False)
    return hist


def get_global_word_features(binary_line, line_height=180):
    vertical_projection = np.sum(binary_line, axis=0)
    word_threshold = 45000
    word_start = None
    words = []
    consecutive_count = 0
    min_consecutive = 15

    for i, value in enumerate(vertical_projection):
        if value < word_threshold:
            consecutive_count = 0
            if word_start is None:
                word_start = i
        elif value >= word_threshold:
            if word_start is not None:
                consecutive_count += 1
                if consecutive_count >= min_consecutive:
                    words.append((word_start, i - min_consecutive + 1))
                    word_start = None
                    consecutive_count = 0

    output_line = binary_line.copy()
    gaps = []
    for i in range(len(words) - 1):
        _, end = words[i]
        start, _ = words[i + 1]
        cv2.line(
            output_line,
            (end, line_height // 2),
            (start, line_height // 2),
            (0, 0, 255),
            2,
        )
        gaps.append(start - end)
    return output_line, gaps, len(words)


def get_zone_features(binary_line):
    height = binary_line.shape[0]
    split_height = height // 3
    zones = [
        binary_line[:split_height],
        binary_line[split_height : 2 * split_height],
        binary_line[2 * split_height :],
    ]
    viz_upper = np.sum(zones[0] == 0)
    viz_middle = np.sum(zones[1] == 0)
    viz_lower = np.sum(zones[2] == 0)
    return viz_upper, viz_middle, viz_lower
