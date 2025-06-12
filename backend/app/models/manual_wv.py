from flask import jsonify
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from ..helpers.manual_wv.feature_extraction.features import compute_chaincode_histogram, compute_hog, compute_slant_angle_histogram, compute_stroke_width_histogram, contour_hinge_pdf, get_exterior_curves, get_global_word_features, get_interior_contours, get_num_of_black_pixels, get_zone_features
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
matplotlib.use('Agg')

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
        binary_line_inv = cv2.bitwise_not(binary_line)
        interior_contours, num_interior_contours, mean_interior_area, _ = get_interior_contours(binary_line_inv)
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
        ]], columns=global_feature_headers)
        lines_data = pd.concat([lines_data, line_data], ignore_index=True)
    lines_data.to_parquet(f"{line_features}/{sample}.parquet", index=False)

def prepare_all_features(sample: str = "test") -> None:
    os.makedirs(all_features, exist_ok=True)

    line_df = pd.read_parquet(f"{line_features}/{sample}.parquet")
    char_df = pd.read_parquet(f"{char_features}/{sample}.parquet")

    char_out_df = pd.DataFrame(
        np.zeros((line_df.shape[0], len(char_df.columns[1:]))), columns=char_df.columns[1:]
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
            char_data = char_rows.drop(columns=["sample"]).sample(n=min(4, len(char_rows)))
            char_out_df.iloc[i] = char_data.mean(axis=0)
    char_out_df = char_out_df.fillna(0)
    char_out_df = char_out_df.add_prefix("e_")
    line_df = pd.concat([line_df, char_out_df], axis=1)
    line_df.to_parquet(f"{all_features}/{sample}.parquet", index=False)

def train_model() -> None:
    for file in os.listdir(f"{FILE_PATH}/../temp/manual_wv/samples"):
        if file.endswith(".png") and file.startswith("sample"):
            img_path = os.path.join(f"{FILE_PATH}/../temp/manual_wv/samples", file)
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
    joblib.dump(scaler, os.path.join(FILE_PATH, "../temp/manual_wv/scaler.pkl"))
    val = scaler.transform(val)
    
    
    model = build_global_autoencoder(all_features_df.shape[1])
    model.load_weights(os.path.join(FILE_PATH, "../weights/pretrained.weights.h5"))
    
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
    
    model.save_weights(os.path.join(FILE_PATH, "../temp/manual_wv/writer.weights.h5"))
    
    y_scores = np.mean(np.square(val - model.predict(val)), axis=1)
    threshold = np.mean(y_scores) + np.std(y_scores)
    
    with open(os.path.join(FILE_PATH, "../temp/manual_wv/threshold.txt"), 'w') as f:
        f.write(str(threshold))

def predict_feature(x, idx, model):
    recon = model.predict(x, verbose=0)
    return np.square(x[:, idx] - recon[:, idx])

def build_feature_error_model(autoencoder, idx: int):
    x_in   = autoencoder.input
    x_hat  = autoencoder.output
    in_i = tf.keras.layers.Lambda(lambda x: x[:, idx:idx+1], output_shape=(1,))(x_in)
    hat_i = tf.keras.layers.Lambda(lambda x: x[:, idx:idx+1], output_shape=(1,))(x_hat)
    diff = tf.keras.layers.Subtract()([in_i, hat_i])
    sq_err = tf.keras.layers.Lambda(lambda x: tf.square(x), output_shape=(1,))(diff)
    return tf.keras.Model(inputs=x_in, outputs=sq_err)

def predict(x, model):
    return np.square(x - model.predict(x, verbose=0))

def process_explanations(test_features: pd.DataFrame, model: tf.keras.Model, scaler, threshold) -> None:
    normal_features = pd.DataFrame()
    for file in os.listdir(all_features):
        if file.endswith(".parquet") and file.startswith("sample"):
            df = pd.read_parquet(os.path.join(all_features, file))
            normal_features = pd.concat([normal_features, df], ignore_index=True)
    normal_features = normal_features.drop(columns=["sample"])
    columns = normal_features.columns
    normal_features = scaler.transform(normal_features)

    test_reconstructed = np.mean(predict(test_features, model), axis=0)
    normal_reconstructed = np.mean(predict(normal_features, model), axis=0)

    plt.figure(figsize=(10, 7))
    plt.plot(test_reconstructed, label="Test Reconstructed Error")
    plt.plot(normal_reconstructed, label="Normal Reconstructed Error")
    plt.xlabel("Features")
    plt.ylabel("Reconstructed Error")
    plt.xticks(ticks=range(len(columns)), labels=columns, rotation=90)
    plt.title(f"Reconstructed Error for Normal (A) and Test (B) Samples\nTest (B) Reconstructed Error: {np.mean(test_reconstructed):.2f}\nNormal (A) Reconstructed Error: {np.mean(normal_reconstructed):.2f}\nPrediction: {'Different Writer' if np.mean(test_reconstructed) > threshold else 'Same Writer'}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(FILE_PATH, "../temp/manual_wv/reconstructed_error.png"))

    pos_shap_values = pd.DataFrame(columns=columns, index=columns, data=np.zeros((len(columns), len(columns))), dtype=float)
    neg_shap_values = pd.DataFrame(columns=columns, index=columns, data=np.zeros((len(columns), len(columns))), dtype=float)
    background_set = normal_features
    for i in tqdm(range(len(columns)), desc="Computing SHAP values"):
        fe_model  = build_feature_error_model(model, i)
        explainer = shap.DeepExplainer(fe_model, background_set)
        shap_val = explainer.shap_values(test_features)
        shap_arr = np.squeeze(shap_val, axis=2)
        pos_shap = np.where(shap_arr > 0, shap_arr, 0)
        neg_shap = np.where(shap_arr < 0, shap_arr, 0)
        pos_shap_values.iloc[i] = pos_shap.mean(axis=0)
        neg_shap_values.iloc[i] = neg_shap.mean(axis=0)
    
    pos_sum = pos_shap_values.sum(axis=1)

    plt.figure(figsize=(15, 10))
    sns.heatmap(pos_shap_values, cmap="coolwarm", fmt=".2f")
    plt.title("SHAP Value Contributions to High Reconstruction Errors")
    plt.xlabel("Input Features")
    plt.ylabel("Target (Reconstructed) Features")
    plt.xticks(
        ticks=np.arange(len(columns)) + 0.5,
        labels=columns,
        rotation=90,
        ha='center'
    )

    plt.yticks(
        ticks=np.arange(len(columns)) + 0.5,
        labels=columns,
        rotation=0,
        va='center'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FILE_PATH, "../temp/manual_wv/heatmap.png"))

    explanation = shap.Explanation(
        values=pos_sum,
        base_values=np.mean(predict(background_set, model)),
        data=np.mean(test_features, axis=0),
        feature_names=columns
    )
    plt.figure(figsize=(15, 10))
    shap.plots.waterfall(explanation, show=False)
    plt.savefig(os.path.join(FILE_PATH, "../temp/manual_wv/waterfall.png"), bbox_inches='tight', pad_inches=0.2)

def perform_manual_writer_verification(img_path: str = "temp/manual_wv/samples/test.png") -> bool:
    preprocess(f"{FILE_PATH}/../{img_path}", sample="test")
    extract_char_features("test")
    extract_line_features("test")
    prepare_all_features("test")
    test_features = pd.read_parquet(f"{all_features}/test.parquet")
    test_features = test_features.drop(columns=["sample"])
    scaler = joblib.load(os.path.join(FILE_PATH, "../temp/manual_wv/scaler.pkl"))
    test_features = scaler.transform(test_features)
    model = build_global_autoencoder(test_features.shape[1])
    model.load_weights(os.path.join(FILE_PATH, "../temp/manual_wv/writer.weights.h5"))
    predict = model.predict(test_features, verbose=0)
    y_scores = np.mean(np.square(test_features - predict), axis=1)
    y_score = np.mean(y_scores)
    with open(os.path.join(FILE_PATH, "../temp/manual_wv/threshold.txt"), 'r') as f:
        threshold = float(f.read().strip())
    y_pred = np.where(np.array(y_score) <= threshold, 0, 1)

    process_explanations(test_features, model, scaler, threshold)
    return "Same Writer" if y_pred == 0 else "Different Writer";
