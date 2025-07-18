import torch

FIXED_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH_ID = "1-CQQMhsN803IJ3qa_wTibVaA2rqy7H3q" #original
# MODEL_PATH_ID = "1Nyq7o8H8n1kuPrpi1PhGcglvwiTrIyHF"
MODEL_PATH = './app/cache/ofsfd/model/ofsfd.pth'
DATA_PATH = './app/cache/ofsfd'
SIZE = 800

# 0e94e1aabf10b825bf31184c86e31816 0e94e1aabf10b825bf31184c86e31816
# 1.8067766427993774 1.8067766427993774