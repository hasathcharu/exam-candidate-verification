from torchvision import transforms
import cv2
from ..utils.constants import DEVICE, FIXED_SIZE

clip_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FIXED_SIZE),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), # these values were used when CLIP was trained
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

def preprocess(image_path):
    input_img = cv2.imread(image_path)
    input_img = cv2.resize(input_img, FIXED_SIZE)

    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    tensor_image = clip_transform(rgb_img)

    if tensor_image.dim() == 3:
        tensor_image = tensor_image.unsqueeze(0)  # [1, 3, 224, 224] CLIP expects batched inputs

    return input_img, tensor_image.to(DEVICE)