from ..helpers.manual_wv.preprocess.preprocess import read_image, remove_rules
from ..helpers.automatic_wv.texture_creation.texture_creation import create_texture
from werkzeug.exceptions import InternalServerError
import os
import matplotlib.pyplot as plt


FILE_PATH = os.path.dirname(__file__)

def preprocess_image(image, save_path=None):
    image = read_image(image)

    if image is None:
        raise InternalServerError("Issues when loading images. Please try again.")
    
    _, binary_image = remove_rules(image)

    if save_path is not None:
        plt.imsave(save_path, binary_image, cmap="gray")

    return binary_image
    
def perform_automatic_writer_verification(sample1_path, sample2_path):
    output_dir = os.path.join(FILE_PATH,"../cache/automatic_wv/binary")
    os.makedirs(output_dir, exist_ok=True)

    binary_image1 = preprocess_image(sample1_path, os.path.join(output_dir, "binary1.png"))
    binary_image2 = preprocess_image(sample2_path, os.path.join(output_dir, "binary2.png"))

    texture_base_dir = os.path.join(FILE_PATH, "../cache/automatic_wv/textures")

    create_texture(
        binary_image1,
        output_dir=os.path.join(texture_base_dir, "image1"),
        prefix="image1"
    )

    create_texture(
        binary_image2,
        output_dir=os.path.join(texture_base_dir, "image2"),
        prefix="image2"
    )
