from ..helpers.manual_wv.preprocess.preprocess import read_image, remove_rules
from ..helpers.automatic_wv.texture_creation.texture_creation import create_texture
from werkzeug.exceptions import InternalServerError
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from keras.saving import register_keras_serializable
from tensorflow.keras import backend as K
import re
import joblib

FILE_PATH = os.path.dirname(__file__)
IMG_SIZE = (224,224)

@register_keras_serializable()
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)

@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, K.epsilon()))

@register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    square_pred = tf.square(y_pred)
    margin = 0.7
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean((1 - y_true) * square_pred + y_true * margin_square)

@register_keras_serializable()
def output_shape(input_shapes):
    return (input_shapes[0][0], 1)

MODEL = tf.keras.models.load_model(
os.path.join(FILE_PATH,"../weights/siamese_model.keras"),
custom_objects={
    "l2_normalize": l2_normalize,
    "euclidean_distance": euclidean_distance,
    "contrastive_loss": contrastive_loss,
    "output_shape": output_shape
},
safe_mode=False
)
FEATURE_EXTRACTOR = MODEL.get_layer("cnn_backbone")

CLF_PATH = os.path.abspath(os.path.join(FILE_PATH, "../weights/logistic_model.joblib"))
clf = joblib.load(CLF_PATH)

def preprocess_image(image, save_path=None):
    image = read_image(image)
    if image is None:
        raise InternalServerError("Issues when loading images. Please try again.")
    _, binary_image = remove_rules(image)
    if save_path is not None:
        plt.imsave(save_path, binary_image, cmap="gray")
    return binary_image

def preprocess_image_for_vgg16(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def compute_embeddings(patches):
    embeddings = []
    for patch in patches:
        input = preprocess_image_for_vgg16(patch)
        embedding = FEATURE_EXTRACTOR.predict(input)
        embeddings.append(embedding[0])
    embeddings = np.stack(embeddings, axis=0)  
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding 

def preprocess_and_create_textures(input_path, binary_output_path, textures_output_dir, prefix):
    binary_image = preprocess_image(input_path, binary_output_path)
    create_texture(
        binary_image,
        output_dir=textures_output_dir,
        prefix=prefix
    )

    def sort_key(path):
        match = re.search(r"_T(\d+)\.png$", path)
        return int(match.group(1)) if match else 0

    patch_files = sorted([
        os.path.join(textures_output_dir, fname)
        for fname in os.listdir(textures_output_dir)
        if fname.endswith(".png")
    ], key=sort_key)
    return patch_files


def perform_automatic_writer_verification(sample1_path, sample2_path):
    output_dir = os.path.join(FILE_PATH,"../cache/automatic_wv/normal/binary")
    os.makedirs(output_dir, exist_ok=True)
    texture_base_dir = os.path.join(FILE_PATH, "../cache/automatic_wv/normal/textures")
    os.makedirs(texture_base_dir, exist_ok=True)
    sample1_patches = preprocess_and_create_textures(
        input_path=sample1_path,
        binary_output_path=os.path.join(output_dir, "binary1.png"),
        textures_output_dir=os.path.join(texture_base_dir, "image1"),
        prefix="image1"
    )
    sample2_patches = preprocess_and_create_textures(
        input_path=sample2_path,
        binary_output_path=os.path.join(output_dir, "binary2.png"),
        textures_output_dir=os.path.join(texture_base_dir, "image2"),
        prefix="image2"
    )
    embedding1 = compute_embeddings(sample1_patches)
    embedding2 = compute_embeddings(sample2_patches)
    distance = np.linalg.norm(embedding1 - embedding2) 
    threshold = 0.7173469662666321
    same_writer = distance <= threshold
    return {
        "distance": float(distance),
        "threshold": threshold,
        "same_writer": bool(same_writer)
    }

def perform_pairwise_automatic_writer_verification(file1_normal_path, file1_fast_path, file2_normal_path, file2_fast_path):
    output_dir = os.path.join(FILE_PATH,"../cache/automatic_wv/pairwise/binary")
    os.makedirs(output_dir, exist_ok=True)
    texture_base_dir = os.path.join(FILE_PATH, "../cache/automatic_wv/pairwise/textures")
    os.makedirs(texture_base_dir, exist_ok=True)
    file1_N_patches = preprocess_and_create_textures(
        input_path=file1_normal_path,
        binary_output_path=os.path.join(output_dir, "file1_n_binary.png"),
        textures_output_dir=os.path.join(texture_base_dir, "file1_N"),
        prefix="file1_N"
    )
    file1_F_patches = preprocess_and_create_textures(
        input_path=file1_fast_path,
        binary_output_path=os.path.join(output_dir, "file1_f_binary.png"),
        textures_output_dir=os.path.join(texture_base_dir, "file1_F"),
        prefix="file1_F"
    )
    file2_N_patches = preprocess_and_create_textures(
        input_path=file2_normal_path,
        binary_output_path=os.path.join(output_dir, "file2_n_binary.png"),
        textures_output_dir=os.path.join(texture_base_dir, "file2_N"),
        prefix="file2_N"
    )
    file2_F_patches = preprocess_and_create_textures(
        input_path=file2_fast_path,
        binary_output_path=os.path.join(output_dir, "file2_f_binary.png"),
        textures_output_dir=os.path.join(texture_base_dir, "file2_F"),
        prefix="file2_F"
    )
    embedding1_N = compute_embeddings(file1_N_patches)
    embedding1_F = compute_embeddings(file1_F_patches)
    embedding2_N = compute_embeddings(file2_N_patches)
    embedding2_F = compute_embeddings(file2_F_patches)
    scores = {
        "1N_vs_1F": np.linalg.norm(embedding1_N - embedding1_F),
        "2N_vs_2F": np.linalg.norm(embedding2_N - embedding2_F),
        "1N_vs_2N": np.linalg.norm(embedding1_N - embedding2_N),
        "1N_vs_2F": np.linalg.norm(embedding1_N - embedding2_F),
        "1F_vs_2N": np.linalg.norm(embedding1_F - embedding2_N),
        "1F_vs_2F": np.linalg.norm(embedding1_F - embedding2_F),
    }
    X_new = np.array([
        [
            scores["1N_vs_1F"],
            scores["2N_vs_2F"],
            scores["1N_vs_2N"],
            scores["1N_vs_2F"],
            scores["1F_vs_2N"],
            scores["1F_vs_2F"],
        ]
    ])
    prob = clf.predict_proba(X_new)[0,1]
    label = clf.predict(X_new)[0]

    return {
        "probability": float(prob),
        "same_writer": bool(label == 0)
    }
        