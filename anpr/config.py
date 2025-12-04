# config.py
import os

# Đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn Model và Label
MODEL_PATH = os.path.join(BASE_DIR, 'anpr_cnn_model.h5')
LABEL_PATH = os.path.join(BASE_DIR, 'label_chars.json')

# Kích thước ảnh cho Model CNN
IMG_WIDTH = 28
IMG_HEIGHT = 28

# Các tham số khác (nếu cần)
DEBUG_MODE = True