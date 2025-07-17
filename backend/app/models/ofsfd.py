import torch.nn as nn
import torch
import gdown
import os
import clip
from ..helpers.ofsfd.utils.constants import DEVICE, MODEL_PATH, MODEL_PATH_ID
import matplotlib.pyplot as plt
from ..helpers.ofsfd.preprocess.preprocess import preprocess

class SignatureVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, _ = clip.load("ViT-B/32", device=DEVICE)
        self.clip.float()

        # Scale the similarity scores. Higher values make output logits more confident.
        self.temperature = nn.Parameter(torch.tensor([1.0], device=DEVICE))

        self.text_descriptions = [
            "a forged signature with irregular pressure, hesitant strokes, unnatural overlap, inconsistent transitions, uneven spacing, unusual curvature, irregular textures, unnatural connectivity.",
            "a genuine signature with consistent pressure, fluid strokes, natural overlap, smooth transitions, balanced spacing, characteristic curvature, structured textures, natural connectivity."

        ]
        self._init_text_features()

    def _init_text_features(self):
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(t) for t in self.text_descriptions]).to(DEVICE)
            text_features = self.clip.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            self.register_buffer('text_features', text_features)

    def forward(self, images):
        images = images.to(device=DEVICE)

        with torch.autocast(device_type=DEVICE):
            image_features = self.clip.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            logits = (image_features @ self.text_features.T) * torch.exp(self.temperature)

        return logits
    
def predict_signature_type(image_path, model_path = MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={MODEL_PATH_ID}", model_path, quiet=False)
        
    model = SignatureVerifier().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    orig_img, tensor_image = preprocess(image_path)

    with torch.no_grad(), torch.autocast(device_type=DEVICE):
        output = model(tensor_image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = "Genuine" if pred == 1 else "Forged"
    confidence = probs[0][pred].item()

    print(f"Prediction: {label} ({confidence:.2f})")

    plt.imshow(orig_img)
    plt.title(f"Prediction: {label} ({confidence:.2f})")
    plt.axis('off')
    plt.savefig(f"{image_path}_prediction.png", bbox_inches='tight', pad_inches=0.1)

    return label, confidence
