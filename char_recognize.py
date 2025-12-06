import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class CharRecognizer:
    def __init__(self, model_path='model/character_recognition.h5'):
        """
        Khởi tạo bộ nhận diện ký tự.
        :param model_path: Đường dẫn tới file model đã train (.h5)
        """
        self.model = None
        self.class_names = self._get_class_names()
        try:
            self.model = load_model(model_path)
            print(f"[INFO] Đã load model nhận diện ký tự từ: {model_path}")
        except Exception as e:
            print(f"[ERROR] Không thể load model: {e}")
            print("[INFO] Vui lòng kiểm tra lại đường dẫn file .h5")

    def _get_class_names(self):
        """
        Định nghĩa map từ chỉ số (index) sang ký tự thực tế.
        Thường dùng cho dataset 36 ký tự (0-9, A-Z).
        """
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {i: char for i, char in enumerate(chars)}

    def preprocess(self, img):
        """
        Tiền xử lý ảnh ký tự cắt ra từ biển số trước khi đưa vào model.
        Đảm bảo kích thước và channel khớp với lúc train (thường là 28x28 hoặc 32x32, Grayscale).
        """
        try:
            # Chuyển về ảnh xám nếu chưa phải
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize về kích thước chuẩn của model (ví dụ 28x28 hoặc 32x32)
            # Bạn cần sửa số này khớp với model của bạn
            target_size = (28, 28) 
            img = cv2.resize(img, target_size)

            # Chuẩn hóa giá trị pixel về [0, 1]
            img = img.astype('float32') / 255.0

            # Mở rộng chiều để khớp với input của model (1, 28, 28, 1)
            img = np.expand_dims(img, axis=-1) # Thêm channel
            img = np.expand_dims(img, axis=0)  # Thêm batch size
            return img
        except Exception as e:
            print(f"[ERROR] Lỗi tiền xử lý ảnh: {e}")
            return None

    def predict(self, char_img):
        """
        Dự đoán ký tự từ ảnh đầu vào.
        :param char_img: Ảnh của một ký tự đã cắt (numpy array)
        :return: Ký tự (str) và độ tin cậy (float)
        """
        if self.model is None:
            return None, 0.0

        processed_img = self.preprocess(char_img)
        if processed_img is None:
            return "", 0.0

        # Dự đoán
        prediction = self.model.predict(processed_img, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)

        predicted_char = self.class_names.get(class_idx, "?")
        
        return predicted_char, confidence

# --- Phần test độc lập (chạy khi gọi trực tiếp file này) ---
if __name__ == "__main__":
    # Giả lập load model và test thử một ảnh
    print("--- Bắt đầu test module ---")
    
    # Cần file model thực tế để chạy
    recognizer = CharRecognizer(model_path='license_plate_model.h5')
    
    # Tạo một ảnh đen giả lập kích thước 30x30
    dummy_img = np.zeros((30, 30, 3), dtype=np.uint8)
    
    char, conf = recognizer.predict(dummy_img)
    print(f"Ký tự dự đoán: {char}, Độ tin cậy: {conf:.2f}")