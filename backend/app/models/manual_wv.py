import math
from flask import json, jsonify
import numpy as np
import shap
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from ..helpers.manual_wv.feature_extraction.features import (
    compute_chaincode_histogram,
    compute_hog,
    compute_slant_angle_histogram,
    compute_stroke_width_histogram,
    contour_hinge_pdf,
    get_exterior_curves,
    get_global_word_features,
    get_interior_contours,
    get_num_of_black_pixels,
    get_zone_features,
)
from ..helpers.manual_wv.preprocess.preprocess import read_image, remove_rules
from ..helpers.manual_wv.preprocess.segmentation import segment_lines, clean_lines
from ..helpers.manual_wv.utils import constants as c
from ..helpers.manual_wv.utils.train_utils import build_global_autoencoder
from werkzeug.exceptions import InternalServerError
import threading
import os
import cv2
import pandas as pd
import joblib
from skimage.measure import shannon_entropy
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import shutil
from openai import OpenAI
import secrets

client = OpenAI()

matplotlib.use("Agg")

FILE_PATH = os.path.dirname(__file__)

lock = threading.Lock()

from PIL import Image
from ultralytics import YOLO

char_model = YOLO(os.path.join(FILE_PATH, "../weights/yolo.pt"))
contour_hinge_pca = joblib.load(
    os.path.join(FILE_PATH, "../weights/contour_hinge_hist_pca.pkl")
)
contour_hinge_hist_scaler = joblib.load(
    os.path.join(FILE_PATH, "../weights/contour_hinge_hist_scaler.pkl")
)

CHAR_CONF_THRESHOLD = 0.5
RESIZE_DIM = (40, 40)

segment_folder = f"{FILE_PATH}/../cache/manual_wv/segment"
feature_folder = f"{FILE_PATH}/../cache/manual_wv/features"
char_features = f"{feature_folder}/chars"
line_features = f"{feature_folder}/line"
contour_hinge_features = f"{feature_folder}/contour_hinge"
all_features = f"{feature_folder}/all"

desc_columns = [
    "Number of Black Pixels",
    "Gray Level Threshold",
    "Entropy of Gray Values",
    "Number of Interior Contours",
    "Mean area of Interior Contours",
    "Number of Exterior Curves",
    "Number of Ultra-Fine Strokes",
    "Number of Very Fine Strokes",
    "Number of Fine Strokes",
    "Number of Medium Strokes",
    "Number of Bold Strokes",
    "Number of Heavy Strokes",
    "Mean Gap Between Words",
    "Standard Deviation of Gap Between Words",
    "Number of Words",
    "Chaincode Histogram Right",
    "Chaincode Histogram Down-Right",
    "Chaincode Histogram Down",
    "Chaincode Histogram Down-Left",
    "Chaincode Histogram Left",
    "Chaincode Histogram Up-Left",
    "Chaincode Histogram Up",
    "Chaincode Histogram Up-Right",
    "Contour-Hinge Principal Component 1",
    "Contour-Hinge Principal Component 2",
    "Contour-Hinge Principal Component 3",
    "Contour-Hinge Principal Component 4",
    "Contour-Hinge Principal Component 5",
    "Contour-Hinge Principal Component 6",
    "Contour-Hinge Principal Component 7",
    "Contour-Hinge Principal Component 8",
    "Contour-Hinge Principal Component 9",
    "Contour-Hinge Principal Component 10",
    "Contour-Hinge Principal Component 11",
    "Contour-Hinge Principal Component 12",
    "Contour-Hinge Principal Component 13",
    "Contour-Hinge Principal Component 14",
    "Contour-Hinge Principal Component 15",
    "Number of Strong Leftward Slant Components",
    "Number of Moderate-Strong Left Slant Components",
    "Number of Moderate Left Slant Components",
    "Number of Slight Left Slant Components",
    "Number of Near Vertical Orientation Components",
    "Number of Slight Right Slant Components",
    "Number of Moderate Right Slant Components",
    "Number of Moderate-Strong Right Slant Components",
    "Number of Strong Rightward Slant Components",
    "Number of Pixels in the Upper Region of the Line",
    "Number of Pixels in the Middle Region of the Line",
    "Number of Pixels in the Lower Region of the Line",
    "Letter 'e' Shape Descriptor 1",
    "Letter 'e' Shape Descriptor 2",
    "Letter 'e' Shape Descriptor 3",
    "Letter 'e' Shape Descriptor 4",
    "Letter 'e' Shape Descriptor 5",
    "Letter 'e' Shape Descriptor 6",
    "Letter 'e' Shape Descriptor 7",
    "Letter 'e' Shape Descriptor 8",
    "Letter 'e' Shape Descriptor 9",
]

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
    "mean_interior_area",
    "num_exterior_curves",
    *[f"stroke_width_hist_{i}" for i in range(6)],
    "mean_word_gap",
    "std_word_gap",
    "num_words",
    *[f"chaincode_hist_{i}" for i in range(8)],
    *[f"contour_hinge_pca_{i}" for i in range(15)],
    *[f"slant_angle_hist_{i}" for i in range(9)],
    "viz_upper",
    "viz_middle",
    "viz_lower",
]


def create_json(
    task_id,
    same_writer=None,
    threshold=None,
    test_reconstructed=None,
    normal_reconstructed=None,
    confidence=None,
    features=None,
    top_5_pos_sum=None,
    top_5_neg_sum=None,
    description=None,
):
    json_data = {}

    if same_writer is not None:
        json_data["same_writer"] = same_writer

    if threshold is not None:
        json_data["threshold"] = threshold

    if test_reconstructed is not None:
        json_data["test_reconstructed"] = test_reconstructed

    if normal_reconstructed is not None:
        json_data["normal_reconstructed"] = normal_reconstructed

    if confidence is not None:
        json_data["confidence"] = confidence

    if features is not None:
        json_data["features"] = features

    if top_5_pos_sum is not None:
        json_data["top_5_pos_sum"] = top_5_pos_sum

    if top_5_neg_sum is not None:
        json_data["top_5_neg_sum"] = top_5_neg_sum

    if description is not None:
        json_data["description"] = description

    json_data["task_id"] = task_id

    return json_data


def save_json(task_id, json_data):
    output_json_path = os.path.join(
        FILE_PATH, f"../rcache/manual_wv/result/{task_id}/result.json"
    )

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=2)


def preprocess(img_path: str, sample: str = "test") -> None:
    image = read_image(img_path)
    if image is None:
        raise InternalServerError(
            "We couldn't load the test image. Please try again from the beginning."
        )
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

        cv2.imwrite(
            os.path.join(f"{segment_folder}/{sample}", "binary.png"), binary_image
        )
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
            resized.save(
                os.path.join(f"{segment_folder}/{sample}/chars/{i}", crop_filename)
            )


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
        char_data = pd.DataFrame(
            [[sample_name, *hog_hist]], columns=local_feature_headers
        )
        chars_data = pd.concat([chars_data, char_data], ignore_index=True)
    chars_data.to_parquet(f"{char_features}/{sample}.parquet", index=False)


def extract_line_features(sample: str = "test") -> None:
    os.makedirs(line_features, exist_ok=True)

    line_images = []
    lines_data = pd.DataFrame(columns=global_feature_headers)

    for line in os.listdir(f"{segment_folder}/{sample}/binary_lines"):
        if line.endswith(".png"):
            line_images.append(
                (
                    os.path.join(f"{segment_folder}/{sample}/raw_lines", line),
                    os.path.join(f"{segment_folder}/{sample}/dirty_binary_lines", line),
                )
            )
    for raw_line_image, dirty_binary_line_image in line_images:
        sample_name = f"{sample}_{os.path.basename(raw_line_image).split('.')[0]}"
        raw_line = cv2.imread(raw_line_image, cv2.IMREAD_GRAYSCALE)
        dirty_binary_line = cv2.imread(dirty_binary_line_image, cv2.IMREAD_GRAYSCALE)

        grey_level_threshold, binary_line = cv2.threshold(
            raw_line, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        grey_entropy = shannon_entropy(raw_line)
        num_black_pixels = get_num_of_black_pixels(binary_line)
        binary_line_inv = cv2.bitwise_not(binary_line)
        interior_contours, num_interior_contours, mean_interior_area, _ = (
            get_interior_contours(binary_line_inv)
        )
        exterior_curves, num_exterior_curves = get_exterior_curves(binary_line_inv)
        chaincode_histogram, chaincode_images, color_chaincode_image = (
            compute_chaincode_histogram(binary_line_inv)
        )
        stroke_width_hist, stroke_widths = compute_stroke_width_histogram(
            binary_line_inv
        )
        gap_line, gaps, num_words = get_global_word_features(binary_line)
        viz_upper, viz_middle, viz_lower = get_zone_features(binary_line)
        contour_hinge_hist = contour_hinge_pdf(binary_line_inv)
        contour_hinge_hist = contour_hinge_hist_scaler.transform([contour_hinge_hist])
        contour_hinge_hist = contour_hinge_pca.transform(contour_hinge_hist)
        contour_hinge_pca_data = contour_hinge_hist.flatten()

        slant_angle_hist = compute_slant_angle_histogram(binary_line_inv)
        line_data = pd.DataFrame(
            [
                [
                    sample_name,
                    num_black_pixels,
                    grey_level_threshold,
                    grey_entropy,
                    num_interior_contours,
                    mean_interior_area,
                    num_exterior_curves,
                    *stroke_width_hist,
                    np.mean(gaps) if len(gaps) > 0 else 0,
                    np.std(gaps) if len(gaps) > 0 else 0,
                    num_words,
                    *chaincode_histogram,
                    *contour_hinge_pca_data,
                    *slant_angle_hist,
                    viz_upper,
                    viz_middle,
                    viz_lower,
                ]
            ],
            columns=global_feature_headers,
        )
        lines_data = pd.concat([lines_data, line_data], ignore_index=True)
    lines_data.to_parquet(f"{line_features}/{sample}.parquet", index=False)


def prepare_all_features(sample: str = "test") -> None:
    os.makedirs(all_features, exist_ok=True)

    line_df = pd.read_parquet(f"{line_features}/{sample}.parquet")
    char_df = pd.read_parquet(f"{char_features}/{sample}.parquet")

    char_out_df = pd.DataFrame(
        np.zeros((line_df.shape[0], len(char_df.columns[1:]))),
        columns=char_df.columns[1:],
    )

    for i, row in line_df.iterrows():
        sample_id = row["sample"]
        char_rows = char_df[char_df["sample"] == sample_id]
        if len(char_rows) > 0:
            char_data = char_rows.drop(columns=["sample"])
            char_out_df.iloc[i] = char_data.mean(axis=0)
        else:
            sample_id = "_".join(sample_id.split("_")[:-1])
            char_rows = char_df[char_df["sample"].str.startswith(sample_id)]
            char_data = char_rows.drop(columns=["sample"]).sample(
                n=min(4, len(char_rows))
            )
            char_out_df.iloc[i] = char_data.mean(axis=0)
    char_out_df = char_out_df.fillna(0)
    char_out_df = char_out_df.add_prefix("e_")
    line_df = pd.concat([line_df, char_out_df], axis=1)
    line_df.to_parquet(f"{all_features}/{sample}.parquet", index=False)


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    auc_roc = auc(fpr, tpr)
    eer_threshold = thresholds[eer_threshold_idx]
    return eer, auc_roc, eer_threshold


def resize_and_save_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise InternalServerError(f"Could not read image at {image_path}")
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 30])


def train_model() -> None:
    for file in os.listdir(f"{FILE_PATH}/../cache/manual_wv/samples"):
        if file.endswith(".png") and file.startswith("sample"):
            img_path = os.path.join(f"{FILE_PATH}/../cache/manual_wv/samples", file)
            preprocess(img_path, sample=file.split(".")[0])
            extract_char_features(file.split(".")[0])
            extract_line_features(file.split(".")[0])
            prepare_all_features(file.split(".")[0])

    all_features_df = pd.DataFrame()
    for file in os.listdir(all_features):
        if file.endswith(".parquet") and file.startswith("sample"):
            df = pd.read_parquet(os.path.join(all_features, file))
            all_features_df = pd.concat([all_features_df, df], ignore_index=True)

    scaler = MinMaxScaler()
    all_features_df = all_features_df.drop(columns=["sample"])
    train, val = train_test_split(all_features_df, test_size=0.2, random_state=42)

    train = scaler.fit_transform(train)
    joblib.dump(scaler, os.path.join(FILE_PATH, "../cache/manual_wv/scaler.pkl"))
    val = scaler.transform(val)

    model = build_global_autoencoder(all_features_df.shape[1])
    # model.load_weights(os.path.join(FILE_PATH, "../weights/pretrained.weights.h5"))

    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    model.fit(
        train,
        train,
        epochs=400,
        batch_size=16,
        shuffle=True,
        verbose=1,
        validation_data=(val, val),
        callbacks=[es],
    )

    model.save_weights(os.path.join(FILE_PATH, "../cache/manual_wv/writer.weights.h5"))

    y_scores = np.mean(np.square(val - model.predict(val)), axis=1)
    val_other = pd.read_parquet(os.path.join(FILE_PATH, "../weights/val_data.parquet"))
    val_other = val_other.sample(n=len(val), random_state=42)
    val_other = val_other.drop(columns=["sample"])
    val_other = scaler.transform(val_other)
    val_all = np.vstack([val, val_other])
    val_reconstructed = model.predict(val_all, verbose=0)
    y_scores = np.mean(np.square(val_all - val_reconstructed), axis=1)
    labels = np.concatenate(
        [np.zeros(len(val), dtype=int), np.ones(len(val_other), dtype=int)]
    )

    max_error = np.max(y_scores)
    _, _, threshold = compute_eer(labels, y_scores)

    json_data = {
        "max_error": max_error,
        "threshold": threshold,
    }

    with open(os.path.join(FILE_PATH, "../cache/manual_wv/threshold.json"), "w") as f:
        json.dump(json_data, f, indent=2)


def predict_feature(x, idx, model):
    recon = model.predict(x, verbose=0)
    return np.square(x[:, idx] - recon[:, idx])


def build_feature_error_model(autoencoder, idx: int):
    x_in = autoencoder.input
    x_hat = autoencoder.output
    in_i = tf.keras.layers.Lambda(lambda x: x[:, idx : idx + 1], output_shape=(1,))(
        x_in
    )
    hat_i = tf.keras.layers.Lambda(lambda x: x[:, idx : idx + 1], output_shape=(1,))(
        x_hat
    )
    diff = tf.keras.layers.Subtract()([in_i, hat_i])
    sq_err = tf.keras.layers.Lambda(lambda x: tf.square(x), output_shape=(1,))(diff)
    return tf.keras.Model(inputs=x_in, outputs=sq_err)


def predict(x, model):
    return np.mean(np.square(x - model.predict(x, verbose=0)), axis=0)


def process_explanations(
    task_id: str,
) -> None:

    scaler = joblib.load(os.path.join(FILE_PATH, "../cache/manual_wv/scaler.pkl"))
    test_features = pd.read_parquet(f"{all_features}/test.parquet")
    test_features = test_features.drop(columns=["sample"])
    test_features_scaled = scaler.transform(test_features)
    model = build_global_autoencoder(test_features_scaled.shape[1])
    model.load_weights(os.path.join(FILE_PATH, "../cache/manual_wv/writer.weights.h5"))
    test_reconstructed = predict(test_features_scaled, model)
    y_score = np.mean(test_reconstructed)
    with open(os.path.join(FILE_PATH, "../cache/manual_wv/threshold.json"), "r") as f:
        data = json.load(f)
        threshold = data["threshold"]
        max_error = data["max_error"]
    y_pred = np.where(np.array(y_score) <= threshold, 0, 1)
    confidence = reconstruction_confidence(y_score, threshold)

    normal_features = pd.DataFrame()
    for file in os.listdir(all_features):
        if file.endswith(".parquet") and file.startswith("sample"):
            df = pd.read_parquet(os.path.join(all_features, file))
            normal_features = pd.concat([normal_features, df], ignore_index=True)
    normal_features = normal_features.drop(columns=["sample"])
    columns = normal_features.columns
    normal_features_scaled = scaler.transform(normal_features)

    test_reconstructed = predict(test_features_scaled, model)
    normal_reconstructed = predict(normal_features_scaled, model)

    plt.figure(figsize=(10, 7))
    plt.plot(test_reconstructed, label="Test Reconstructed Error")
    plt.plot(normal_reconstructed, label="Normal Reconstructed Error")
    plt.xlabel("Features")
    plt.ylabel("Reconstructed Error")
    plt.xticks(ticks=range(len(columns)), labels=desc_columns, rotation=90)
    plt.title(
        f"Reconstructed Error for Normal and Test Samples\nTest Reconstructed Error: {np.mean(test_reconstructed):.2f}\nNormal Reconstructed Error: {np.mean(normal_reconstructed):.2f}\nPrediction: {'Different Writer' if np.mean(test_reconstructed) > threshold else 'Same Writer'}"
    )
    plt.tight_layout()
    plt.legend()
    plt.savefig(
        os.path.join(
            FILE_PATH, f"../rcache/manual_wv/result/{task_id}/reconstructed_error.png"
        )
    )

    print(
        "Prediction\t: ",
        "\033[91m" + ("Same Writer" if y_pred == 0 else "Different Writer") + "\033[0m",
    )
    print("Score\t\t: ", "\033[91m" + str(y_score) + "\033[0m")
    print("Threshold\t: ", "\033[91m" + str(threshold) + "\033[0m")
    print("Confidence\t: ", "\033[91m" + str(confidence) + "\033[0m")

    pos_shap_values = pd.DataFrame(
        columns=desc_columns,
        index=desc_columns,
        data=np.zeros((len(columns), len(columns))),
        dtype=float,
    )

    neg_shap_values = pd.DataFrame(
        columns=desc_columns,
        index=desc_columns,
        data=np.zeros((len(columns), len(columns))),
        dtype=float,
    )

    background_set = normal_features_scaled
    if os.environ.get("EXP_ENABLED") == "true":
        for i in tqdm(range(len(columns)), desc="Computing SHAP values"):
            fe_model = build_feature_error_model(model, i)
            explainer = shap.DeepExplainer(fe_model, background_set)
            shap_val = explainer.shap_values(test_features_scaled)
            shap_arr = np.squeeze(shap_val, axis=2)
            pos_shap = np.where(shap_arr > 0, shap_arr, 0)
            neg_shap = np.where(shap_arr < 0, shap_arr, 0)
            pos_shap_values.iloc[i] = pos_shap.mean(axis=0)
            neg_shap_values.iloc[i] = neg_shap.mean(axis=0)

    pos_sum = pos_shap_values.sum(axis=1)
    neg_sum = neg_shap_values.sum(axis=1)

    plt.figure(figsize=(15, 10))
    sns.heatmap(pos_shap_values, cmap="coolwarm", fmt=".2f")
    plt.title("SHAP Value Contributions to High Reconstruction Errors")
    plt.xlabel("Input Features")
    plt.ylabel("Target (Reconstructed) Features")
    plt.xticks(
        ticks=np.arange(len(columns)) + 0.5,
        labels=desc_columns,
        rotation=90,
        ha="center",
    )

    plt.yticks(
        ticks=np.arange(len(columns)) + 0.5,
        labels=desc_columns,
        rotation=0,
        va="center",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}/pos_heatmap.png")
    )

    plt.figure(figsize=(15, 10))
    sns.heatmap(np.abs(neg_shap_values), cmap="coolwarm", fmt=".2f")
    plt.title("SHAP Value Contributions to Low Reconstruction Errors")
    plt.xlabel("Input Features")
    plt.ylabel("Target (Reconstructed) Features")
    plt.xticks(
        ticks=np.arange(len(columns)) + 0.5,
        labels=desc_columns,
        rotation=90,
        ha="center",
    )

    plt.yticks(
        ticks=np.arange(len(columns)) + 0.5,
        labels=desc_columns,
        rotation=0,
        va="center",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}/neg_heatmap.png")
    )

    explanation = shap.Explanation(
        values=pos_sum,
        base_values=np.mean(predict(background_set, model)),
        data=np.mean(test_features_scaled, axis=0),
        feature_names=desc_columns,
    )
    plt.figure(figsize=(15, 10))
    shap.plots.waterfall(explanation, show=False)
    plt.savefig(
        os.path.join(
            FILE_PATH, f"../rcache/manual_wv/result/{task_id}/pos_waterfall.png"
        ),
        bbox_inches="tight",
        pad_inches=0.2,
    )

    explanation = shap.Explanation(
        values=neg_sum,
        base_values=np.mean(predict(background_set, model)),
        data=np.mean(test_features_scaled, axis=0),
        feature_names=desc_columns,
    )
    plt.figure(figsize=(15, 10))
    shap.plots.waterfall(explanation, show=False)
    plt.savefig(
        os.path.join(
            FILE_PATH, f"../rcache/manual_wv/result/{task_id}/neg_waterfall.png"
        ),
        bbox_inches="tight",
        pad_inches=0.2,
    )

    resize_and_save_image(
        os.path.join(FILE_PATH, "../cache/manual_wv/samples/sample1.png"),
        os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}/known.png"),
    )

    resize_and_save_image(
        os.path.join(FILE_PATH, "../cache/manual_wv/samples/test.png"),
        os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}/test.png"),
    )

    shutil.copy(
        os.path.join(FILE_PATH, "../cache/ofsfd/samples/test.png"),
        os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}/test-sig.png"),
    )

    top_5_pos = pos_sum.sort_values(ascending=False).head(5)
    top_5_neg = neg_sum.sort_values(ascending=True).head(5)

    combined = [
        {"feature": name, "normal": n, "test": t}
        for name, n, t in zip(
            desc_columns, normal_features.mean(axis=0), test_features.mean(axis=0)
        )
    ]

    save_json(
        task_id,
        create_json(
            task_id=task_id,
            same_writer=((1 - y_pred) == 1).tolist(),
            threshold=threshold,
            test_reconstructed=np.mean(test_reconstructed).item(),
            normal_reconstructed=np.mean(normal_reconstructed).item(),
            confidence=confidence,
            features=combined,
            top_5_pos_sum=dict(top_5_pos.sort_values(ascending=False).items()),
            top_5_neg_sum=dict(top_5_neg.sort_values(ascending=False).items()),
        ),
    )


def process_interpretation(
    task_id: str,
) -> None:
    with open(
        os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}/result.json"),
        "r",
    ) as f:
        json_data = json.load(f)

    with open(os.path.join(FILE_PATH, "prompts/sys_intructions.txt"), "r") as f:
        system_instructions = f.read()
    with open(os.path.join(FILE_PATH, "prompts/chat.txt"), "r") as f:
        prompt = f.read()
    with open(os.path.join(FILE_PATH, "prompts/diff_input.json"), "r") as f:
        diff_input = json.load(f)
    with open(os.path.join(FILE_PATH, "prompts/same_input.json"), "r") as f:
        same_input = json.load(f)
    with open(os.path.join(FILE_PATH, "prompts/diff_exp.txt"), "r") as f:
        diff_exp = f.read()
    with open(os.path.join(FILE_PATH, "prompts/same_exp.txt"), "r") as f:
        same_exp = f.read()

    messages = [
        {"role": "system", "content": system_instructions},
        {
            "role": "user",
            "content": prompt.format(json=json.dumps(diff_input, indent=2)),
        },
        {"role": "assistant", "content": diff_exp},
        {
            "role": "user",
            "content": prompt.format(json=json.dumps(same_input, indent=2)),
        },
        {"role": "assistant", "content": same_exp},
        {
            "role": "user",
            "content": prompt.format(json=json.dumps(json_data, indent=2)),
        },
    ]

    description = "Failed to generate the AI explanation due to an error."
    try:
        if (
            os.environ.get("EXP_ENABLED") != "true"
            or os.environ.get("GEN_EXPLANATION") != "true"
        ):
            raise Exception("Explanation generation is disabled.")
        explanation = client.responses.create(
            model="o3",
            reasoning={"effort": "medium"},
            input=messages,
        )
        description = explanation.output_text
    except Exception as e:
        print("Failed to generate content: %s", e)
    json_data["description"] = description
    save_json(task_id, create_json(**json_data))


def reconstruction_confidence(error, threshold, sharpness=15):
    sigmoid = lambda x: 1 / (1 + math.exp(-sharpness * abs(x)))
    confidence = sigmoid(error - threshold)
    return confidence


def create_personalized_verification_task(
    quick_result,
    img_path: str = "cache/manual_wv/samples/test.png",
) -> bool:
    preprocess(f"{FILE_PATH}/../{img_path}", sample="test")
    extract_char_features("test")
    extract_line_features("test")
    prepare_all_features("test")
    task_id = None
    while True:
        try:
            task_id = secrets.token_urlsafe(8)
            os.makedirs(
                os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}/")
            )
            quick_result.save(
                f"{FILE_PATH}/../rcache/manual_wv/result/{task_id}/quick_result.json"
            )
            break
        except FileExistsError:
            continue
    return {
        "task_id": task_id,
    }


def create_personalized_verification_explanation(
    task_id: str,
) -> bool:
    if not os.path.exists(
        os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}")
    ):
        raise InternalServerError(
            "The task ID does not exist. Please create a new task."
        )
    process_explanations(task_id)
    return {
        "task_id": task_id,
    }


def create_personalized_verification_interpretation(
    task_id: str,
) -> bool:
    if not os.path.exists(
        os.path.join(FILE_PATH, f"../rcache/manual_wv/result/{task_id}")
    ):
        raise InternalServerError(
            "The task ID does not exist. Please create a new task."
        )
    process_interpretation(task_id)
    return {
        "task_id": task_id,
    }
