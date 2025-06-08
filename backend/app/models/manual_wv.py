import numpy as np
from ..helpers.manual_wv.feature_extraction.features import compute_chaincode_histogram, compute_hog, compute_slant_angle_histogram, compute_stroke_width_histogram, contour_hinge_pdf, get_exterior_curves, get_global_word_features, get_interior_contours, get_num_of_black_pixels, get_zone_features
from ..helpers.manual_wv.preprocess.preprocess import read_image, remove_rules
from ..helpers.manual_wv.preprocess.segmentation import segment_lines, clean_lines
from ..helpers.manual_wv.utils import constants as c
from werkzeug.exceptions import InternalServerError
import threading
import os
import cv2
import pandas as pd
import joblib
from skimage.measure import shannon_entropy

FILE_PATH = os.path.dirname(__file__)

lock = threading.Lock()

from PIL import Image
from ultralytics import YOLO

char_model = YOLO(os.path.join(FILE_PATH, "../weights/yolo.pt"))
contour_hinge_pca = joblib.load(os.path.join(FILE_PATH, "../weights/contour_hinge_hist_pca.pkl"))
contour_hinge_hist_scaler = joblib.load(os.path.join(FILE_PATH, "../weights/contour_hinge_hist_scaler.pkl"))

CHAR_CONF_THRESHOLD = 0.5
RESIZE_DIM = (40, 40)

segment_folder = f"{FILE_PATH}/../temp/manual_wv/segment"
feature_folder = f"{FILE_PATH}/../temp/manual_wv/features"
char_features = f"{feature_folder}/chars"
line_features = f"{feature_folder}/line"
contour_hinge_features = f"{feature_folder}/contour_hinge"
all_features = f"{feature_folder}/all"

local_feature_headers = [
    "sample",
    *[f"hog_{i}" for i in range(9)],
]

global_feature_headers = [
    "sample",
    "num_black_pixels",
    "grey_level_threshold",
    "grey_entropy",
    "num_interior_contours",
    "num_exterior_curves",
    *[f"stroke_width_hist_{i}" for i in range(6)],
    "mean_word_gap",
    "std_word_gap",
    "num_words",
    "line_overlap",
    *[f"chaincode_hist_{i}" for i in range(8)],
    *[f"contour_hinge_pca_{i}" for i in range(15)],
    *[f"slant_angle_hist_{i}" for i in range(9)],
    "viz_upper",
    "viz_middle",
    "viz_lower",
]

def preprocess(img_path: str, sample: str = "test") -> None:
    image = read_image(img_path)
    if image is None:
        raise InternalServerError("We couldn't load the test image. Please try again from the beginning.")
    raw_image, binary_image = remove_rules(image)
    (lines, dirty_lines), cropped_rectangles = segment_lines(binary_image)
    dirty_raw_lines = [
        raw_image[y1:y2, x1:x2] for (x1, y1, x2, y2) in cropped_rectangles
    ]
    raw_lines, _ = clean_lines(dirty_raw_lines)
    pil_lines = [Image.fromarray(line) for line in lines]

    with lock:
        os.makedirs(f"{segment_folder}/{sample}", exist_ok=True)

        os.makedirs(f"{segment_folder}/{sample}/binary_lines", exist_ok=True)
        os.makedirs(f"{segment_folder}/{sample}/raw_lines", exist_ok=True)
        os.makedirs(f"{segment_folder}/{sample}/dirty_binary_lines", exist_ok=True)
        os.makedirs(f"{segment_folder}/{sample}/dirty_raw_lines", exist_ok=True)

        cv2.imwrite(os.path.join(f"{segment_folder}/{sample}", "binary.png"), binary_image)
        cv2.imwrite(os.path.join(f"{segment_folder}/{sample}", "raw.png"), raw_image)

        for i, line in enumerate(lines):
            cv2.imwrite(f"{segment_folder}/{sample}/binary_lines/{i}.png", line)
        for i, line in enumerate(raw_lines):
            cv2.imwrite(f"{segment_folder}/{sample}/raw_lines/{i}.png", line)
        for i, line in enumerate(dirty_lines):
            cv2.imwrite(f"{segment_folder}/{sample}/dirty_binary_lines/{i}.png", line)
        for i, line in enumerate(dirty_raw_lines):
            cv2.imwrite(f"{segment_folder}/{sample}/dirty_raw_lines/{i}.png", line)

    chars = []

    for i, pil_line in enumerate(pil_lines):
        char_coords = char_model(pil_line, conf=CHAR_CONF_THRESHOLD)[0]
        chars.append([])
        for j, box in enumerate(char_coords.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            char = pil_line.crop((x1, y1, x2, y2))
            resized = char.resize(RESIZE_DIM, Image.LANCZOS)
            crop_filename = f"{sample}_{i}_{j}.png"
            os.makedirs(f"{segment_folder}/{sample}/chars/{i}", exist_ok=True)
            resized.save(os.path.join(f"{segment_folder}/{sample}/chars/{i}", crop_filename))

def extract_char_features(sample: str = "test") -> None:
    os.makedirs(char_features, exist_ok=True)

    char_images = []
    chars_data = pd.DataFrame(columns=local_feature_headers)
    for line in os.listdir(f"{segment_folder}/{sample}/chars"):
            if os.path.isdir(f"{segment_folder}/{sample}/chars/{line}"):
                char_images.extend(
                    [
                        os.path.join(f"{segment_folder}/{sample}/chars/{line}", img)
                        for img in os.listdir(f"{segment_folder}/{sample}/chars/{line}")
                        if img.endswith(".png")
                    ]
                )
    for char_image in char_images:
        char = cv2.imread(char_image, cv2.IMREAD_GRAYSCALE)
        sample_name = os.path.basename(char_image).split(".")[0]
        hog_hist = compute_hog(char)
        char_data = pd.DataFrame([[sample_name, *hog_hist]], columns=local_feature_headers)
        chars_data = pd.concat([chars_data, char_data], ignore_index=True)
    chars_data.to_parquet(f"{char_features}/{sample}.parquet", index=False)

def extract_line_features(sample: str = "test") -> None:
    os.makedirs(line_features, exist_ok=True)

    line_images = []
    lines_data = pd.DataFrame(columns=global_feature_headers)

    for line in os.listdir(f"{segment_folder}/{sample}/binary_lines"):
        if line.endswith(".png"):
            line_images.append((os.path.join(f"{segment_folder}/{sample}/raw_lines", line), os.path.join(f"{segment_folder}/{sample}/dirty_binary_lines", line)))
    for raw_line_image, dirty_binary_line_image in line_images:
        sample_name = f"{sample}_{os.path.basename(raw_line_image).split('.')[0]}"
        raw_line = cv2.imread(raw_line_image, cv2.IMREAD_GRAYSCALE)
        dirty_binary_line = cv2.imread(dirty_binary_line_image, cv2.IMREAD_GRAYSCALE)

        grey_level_threshold, binary_line = cv2.threshold(
            raw_line, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        grey_entropy = shannon_entropy(raw_line)
        num_black_pixels = get_num_of_black_pixels(binary_line)
        line_overlap = get_num_of_black_pixels(dirty_binary_line) - num_black_pixels
        binary_line_inv = cv2.bitwise_not(binary_line)
        interior_contours, num_interior_contours = get_interior_contours(binary_line_inv)
        exterior_curves, num_exterior_curves = get_exterior_curves(binary_line_inv)
        chaincode_histogram, chaincode_images, color_chaincode_image = (
            compute_chaincode_histogram(binary_line_inv)
        )
        stroke_width_hist, stroke_widths = compute_stroke_width_histogram(binary_line_inv)
        gap_line, gaps, num_words = get_global_word_features(binary_line)
        viz_upper, viz_middle, viz_lower = get_zone_features(binary_line)
        contour_hinge_hist = contour_hinge_pdf(binary_line_inv)
        contour_hinge_hist = contour_hinge_hist_scaler.transform([contour_hinge_hist])
        contour_hinge_hist = contour_hinge_pca.transform(contour_hinge_hist)
        contour_hinge_pca_data = contour_hinge_hist.flatten()
        
        slant_angle_hist = compute_slant_angle_histogram(binary_line_inv)
        line_data = pd.DataFrame([[
            sample_name,
            num_black_pixels,
            grey_level_threshold,
            grey_entropy,
            num_interior_contours,
            num_exterior_curves,
            *stroke_width_hist,
            np.mean(gaps) if len(gaps) > 0 else 0,
            np.std(gaps) if len(gaps) > 0 else 0,
            num_words,
            line_overlap,
            *chaincode_histogram,
            *contour_hinge_pca_data,
            *slant_angle_hist,
            viz_upper,
            viz_middle,
            viz_lower,
        ]], columns=global_feature_headers)
        lines_data = pd.concat([lines_data, line_data], ignore_index=True)
    lines_data.to_parquet(f"{line_features}/{sample}.parquet", index=False)

def perform_manual_writer_verification(img_path: str = "temp/manual_wv/samples/test.png") -> bool:
    preprocess(f"{FILE_PATH}/../{img_path}", sample="test")
    extract_char_features("test")
    extract_line_features("test")
    return True;
