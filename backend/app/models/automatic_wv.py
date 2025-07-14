from ..helpers.manual_wv.preprocess.preprocess import read_image, remove_rules
from ..helpers.automatic_wv.texture_creation.texture_creation import create_texture
from werkzeug.exceptions import InternalServerError
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Lambda, GlobalAveragePooling2D
from keras.saving import register_keras_serializable
from tensorflow.keras.models import Model
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

def perform_automatic_writer_verification(sample1_path, sample2_path):
    output_dir = os.path.join(FILE_PATH,"../cache/automatic_wv/binary")
    os.makedirs(output_dir, exist_ok=True)

    binary_image1 = preprocess_image(sample1_path, os.path.join(output_dir, "binary1.png"))
    binary_image2 = preprocess_image(sample2_path, os.path.join(output_dir, "binary2.png"))

    texture_base_dir = os.path.join(FILE_PATH, "../cache/automatic_wv/textures")

    patches1 = create_texture(
        binary_image1,
        output_dir=os.path.join(texture_base_dir, "image1"),
        prefix="image1"
    )

    patches2 = create_texture(
        binary_image2,
        output_dir=os.path.join(texture_base_dir, "image2"),
        prefix="image2"
    )

    patch1 = os.path.join(texture_base_dir, "image1", "image1_T1.png")
    patch2 = os.path.join(texture_base_dir, "image2", "image2_T9.png")

    input1 = preprocess_image_for_vgg16(patch1)
    input2 = preprocess_image_for_vgg16(patch2)

    distance = MODEL.predict([input1, input2])[0][0]  

    threshold = 0.7173469662666321
    same_writer = distance <= threshold

    return {
        "distance": float(distance),
        "threshold": threshold,
        "same_writer": same_writer
    }
