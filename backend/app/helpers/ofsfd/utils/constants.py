import torch

FIXED_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH_ID = "1-CQQMhsN803IJ3qa_wTibVaA2rqy7H3q"
MODEL_PATH = './app/weights/ofsfd.pth'
DATA_PATH = './app/cache/ofsfd'
SIZE = 800
