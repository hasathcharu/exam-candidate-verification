from ..helpers.manual_wv.preprocess.preprocess import read_image, remove_rules
from ..helpers.manual_wv.preprocess.segmentation import segment_lines, clean_lines
from ..helpers.manual_wv.utils import constants as c
from werkzeug.exceptions import InternalServerError
from flask import current_app
import threading
import os
import cv2

FILE_PATH = os.path.dirname(__file__)

lock = threading.Lock()

from PIL import Image
from ultralytics import YOLO

char_model = YOLO(os.path.join(FILE_PATH, "../weights/yolo.pt"))
CHAR_CONF_THRESHOLD = 0.5
RESIZE_DIM = (40, 40)

output_folder = f"{FILE_PATH}/../temp/manual_wv/temp_writer"

def preprocess(img_path, type: str = "test") -> None:
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
        os.makedirs(f"{output_folder}/{type}", exist_ok=True)

        os.makedirs(f"{output_folder}/{type}/binary_lines", exist_ok=True)
        os.makedirs(f"{output_folder}/{type}/raw_lines", exist_ok=True)
        os.makedirs(f"{output_folder}/{type}/dirty_binary_lines", exist_ok=True)
        os.makedirs(f"{output_folder}/{type}/dirty_raw_lines", exist_ok=True)

        cv2.imwrite(os.path.join(f"{output_folder}/{type}", "binary.png"), binary_image)
        cv2.imwrite(os.path.join(f"{output_folder}/{type}", "raw.png"), raw_image)

        for i, line in enumerate(lines):
            cv2.imwrite(f"{output_folder}/{type}/binary_lines/{i}.png", line)
        for i, line in enumerate(raw_lines):
            cv2.imwrite(f"{output_folder}/{type}/raw_lines/{i}.png", line)
        for i, line in enumerate(dirty_lines):
            cv2.imwrite(f"{output_folder}/{type}/dirty_binary_lines/{i}.png", line)
        for i, line in enumerate(dirty_raw_lines):
            cv2.imwrite(f"{output_folder}/{type}/dirty_raw_lines/{i}.png", line)

    chars = []

    for i, pil_line in enumerate(pil_lines):
        char_coords = char_model(pil_line, conf=CHAR_CONF_THRESHOLD)[0]
        chars.append([])
        for j, box in enumerate(char_coords.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            char = pil_line.crop((x1, y1, x2, y2))
            resized = char.resize(RESIZE_DIM, Image.LANCZOS)
            crop_filename = f"{type}_{i}_{j}.png"
            os.makedirs(f"{output_folder}/{type}/chars/{i}", exist_ok=True)
            resized.save(os.path.join(f"{output_folder}/{type}/chars/{i}", crop_filename))

def perform_manual_writer_verification(img_path: str = "temp/manual_test.png") -> bool:
    preprocess(f"{FILE_PATH}/../{img_path}", type="test")
    return True;
