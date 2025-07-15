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

    patch_files = [
        os.path.join(textures_output_dir, fname)
        for fname in os.listdir(textures_output_dir)
        if fname.endswith(".png")
    ]
    return patch_files


def perform_automatic_writer_verification(sample1_path, sample2_path):
    output_dir = os.path.join(FILE_PATH,"../cache/automatic_wv/binary")
    os.makedirs(output_dir, exist_ok=True)

    texture_base_dir = os.path.join(FILE_PATH, "../cache/automatic_wv/textures")
    os.makedirs(texture_base_dir, exist_ok=True)

    patch1_files = preprocess_and_create_textures(
        input_path=sample1_path,
        binary_output_path=os.path.join(output_dir, "binary1.png"),
        textures_output_dir=os.path.join(texture_base_dir, "image1"),
        prefix="image1"
    )

    patch2_files = preprocess_and_create_textures(
        input_path=sample2_path,
        binary_output_path=os.path.join(output_dir, "binary2.png"),
        textures_output_dir=os.path.join(texture_base_dir, "image2"),
        prefix="image2"
    )

    embedding1 = compute_embeddings(patch1_files)
    embedding2 = compute_embeddings(patch2_files)

    distance = np.linalg.norm(embedding1 - embedding2) 

    threshold = 0.7173469662666321
    same_writer = distance <= threshold

    return {
        "distance": float(distance),
        "threshold": threshold,
        "same_writer": bool(same_writer)
    }
