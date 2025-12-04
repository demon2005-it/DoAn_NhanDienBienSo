import cv2
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import "bộ não" CNN rỗng từ file kia
from char_recognize import build_cnn_model

# --- CÁC THAM SỐ ---
DATASET_DIR = 'Character dataset' # Tên thư mục bạn vừa giải nén
LABEL_FILE = 'label_chars.json'
MODEL_OUTPUT = 'anpr_cnn_model.h5' # Tên của "bộ não" đã huấn luyện

# Kích thước ảnh mà CNN yêu cầu (phải khớp với char_recognize.py)
IMG_WIDTH = 28
IMG_HEIGHT = 28

def load_dataset(dataset_dir, label_mapping):
    """
    Tải tất cả ảnh ký tự từ các thư mục con.
    """
    data = []
    labels = []
    
    print(f"[INFO] Bắt đầu tải dữ liệu từ '{dataset_dir}'...")
    
    # Lấy danh sách các thư mục con (chính là tên ký tự)
    char_folders = os.listdir(dataset_dir)
    
    for char_name in char_folders:
        # Kiểm tra xem ký tự này có trong file JSON của chúng ta không
        if char_name not in label_mapping:
            print(f"[CẢNH BÁO] Bỏ qua thư mục không có trong JSON: {char_name}")
            continue
            
        char_label = label_mapping[char_name]
        char_path = os.path.join(dataset_dir, char_name)
        
        # Bỏ qua nếu không phải là thư mục
        if not os.path.isdir(char_path):
            continue
            
        # Đọc từng file ảnh trong thư mục ký tự
        for img_file in os.listdir(char_path):
            img_path = os.path.join(char_path, img_file)
            
            try:
                # Đọc ảnh, chuyển sang ảnh xám
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[LỖI] Không thể đọc ảnh: {img_path}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize ảnh về kích thước chuẩn của CNN (28x28)
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                data.append(img_resized)
                labels.append(char_label)
                
            except Exception as e:
                print(f"Lỗi khi xử lý file {img_path}: {e}")

    print(f"[INFO] Tải xong! Tổng cộng {len(data)} ảnh.")
    return (np.array(data), np.array(labels))

# --- CHƯƠNG TRÌNH CHÍNH ---

# 1. Tải file JSON ánh xạ ký tự
with open(LABEL_FILE, 'r') as f:
    label_mapping = json.load(f)

# Đếm số lượng lớp (ví dụ: 30 ký tự)
NUM_CLASSES = len(label_mapping)

# 2. Tải dữ liệu ảnh
(data, labels) = load_dataset(DATASET_DIR, label_mapping)

# 3. Tiền xử lý dữ liệu cho CNN
# Thêm 1 chiều (channel) cho ảnh xám (VD: 28x28 -> 28x28x1)
data = data.reshape(data.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
# Chuẩn hóa pixel về khoảng [0, 1]
data = data.astype('float32') / 255.0

# 4. Chuyển đổi nhãn (Labels)
# Ví dụ: nhãn '5' -> [0, 0, 0, 0, 0, 1, 0, ...] (One-hot encoding)
labels_categorical = to_categorical(labels, num_classes=NUM_CLASSES)

# 5. Chia dữ liệu
# 80% để huấn luyện (train), 20% để kiểm tra (validation)
(X_train, X_val, Y_train, Y_val) = train_test_split(
    data, labels_categorical, test_size=0.20, random_state=42, stratify=labels_categorical)

print(f"[INFO] Dữ liệu huấn luyện: {X_train.shape[0]} ảnh")
print(f"[INFO] Dữ liệu kiểm tra: {X_val.shape[0]} ảnh")

# 6. Xây dựng mô hình CNN
print("[INFO] Xây dựng kiến trúc mô hình...")
model = build_cnn_model(num_classes=NUM_CLASSES)
model.summary()

# 7. Huấn luyện (Train) mô hình
print("[INFO] Bắt đầu huấn luyện. Việc này có thể mất vài phút...")

# Callbacks: Dừng sớm nếu không cải thiện và chỉ lưu model tốt nhất
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath=MODEL_OUTPUT, monitor='val_loss', save_best_only=True)
]

# Chạy huấn luyện!
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=32,
    epochs=50, # Huấn luyện tối đa 50 vòng
    callbacks=callbacks,
    verbose=1
)

print(f"[INFO] Huấn luyện hoàn tất! Đã lưu mô hình tốt nhất tại '{MODEL_OUTPUT}'")
print("BƯỚC TIẾP THEO: Cập nhật file 'find_plate.py' để SỬ DỤNG mô hình này.")