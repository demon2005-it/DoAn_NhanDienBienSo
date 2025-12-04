# core.py
import cv2
import imutils
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
import config
import utils

class ANPRSystem:
    def __init__(self):
        self.model = None
        self.reverse_label_mapping = {}
        self._load_resources()

    def _load_resources(self):
        print(f"[INFO] Đang tải model từ: {config.MODEL_PATH}")
        try:
            if os.path.exists(config.MODEL_PATH) and os.path.exists(config.LABEL_PATH):
                self.model = load_model(config.MODEL_PATH)
                with open(config.LABEL_PATH, 'r') as f:
                    label_mapping = json.load(f)
                self.reverse_label_mapping = {v: k for k, v in label_mapping.items()}
                print("[INFO] Tải model thành công!")
            else:
                print("[CẢNH BÁO] Không tìm thấy file model hoặc json.")
        except Exception as e:
            print(f"[LỖI] {str(e)}")

    def predict(self, input_image):
        if self.model is None:
            return input_image, "Lỗi: Chưa tải được Model"

        image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        original_image = image.copy()
        
        # --- Bước 1: Tìm biển số ---
        image_resized = imutils.resize(image, width=600)
        ratio = original_image.shape[1] / float(image_resized.shape[1])
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 200) 
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        screenCnt = None
        plate_type = ""
        
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspectRatio = w / float(h)
                if 2.5 <= aspectRatio <= 7.0:
                    screenCnt = approx
                    plate_type = "long"
                    break
                elif 1.0 <= aspectRatio < 2.5:
                    screenCnt = approx
                    plate_type = "square"
                    break

        if screenCnt is None:
            return input_image, "Không tìm thấy biển số xe!"

        # --- Bước 2: Cắt biển số ---
        pts = utils.order_points(screenCnt)
        src_pts = pts * ratio 
        dst_w, dst_h = (450, 150) if plate_type == "long" else (280, 200)
        
        dst_pts = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(original_image, M, (dst_w, dst_h))
        
        cv2.drawContours(original_image, [screenCnt.astype("int") * int(ratio)], -1, (0, 255, 0), 3)

        # --- Bước 3: Xử lý nhị phân ---
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        (_, binary) = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # [FIX 1] Xóa viền
        h_bin, w_bin = binary.shape
        margin = 4
        binary[0:margin, :] = 0
        binary[h_bin-margin:, :] = 0
        binary[:, 0:margin] = 0
        binary[:, w_bin-margin:] = 0
        
        self.result_debug_img = warped.copy() 

        # --- Bước 4: Logic nhận diện ký tự (Đã tách ra hàm riêng bên dưới để gọn) ---
        if plate_type == "square":
            split_y = utils.find_split_line(binary)
            cv2.line(self.result_debug_img, (0, split_y), (dst_w, split_y), (0, 255, 255), 2)
            t1 = self._process_contours(binary[0:split_y-3, :], plate_type, offset_y=0, is_row1=True)
            t2 = self._process_contours(binary[split_y+3:, :], plate_type, offset_y=split_y+3, is_row1=False)
            text_result = f"{t1} - {t2}"
        else: 
            text_result = self._process_contours(binary, plate_type, offset_y=0, is_row1=True)

        final_display = cv2.cvtColor(self.result_debug_img, cv2.COLOR_BGR2RGB)
        return final_display, text_result

    def _process_contours(self, img_binary, plate_type, offset_y=0, is_row1=False):
        local_text = ""
        cnts_char, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_h, roi_w = img_binary.shape

        if len(cnts_char) > 0:
            cnts_sorted, _ = utils.sort_contours_left_to_right(cnts_char)
            idx = 0
            for c in cnts_sorted:
                (x, y, w, h) = cv2.boundingRect(c)
                height_ratio = h / float(roi_h)
                
                # [FIX 2] Điều kiện lọc
                if (h >= 20) and (w / h < 2.0) and (0.3 < height_ratio < 0.95) and (x + w <= roi_w - 3):
                    roi = img_binary[y:y+h, x:x+w]
                    roi_cnn = utils.preprocess_char(roi)
                    
                    if roi_cnn is not None:
                        pred = self.model.predict(roi_cnn, verbose=0)
                        char = self.reverse_label_mapping.get(np.argmax(pred), '?')
                        
                        # Logic Force Number
                        force_number = False
                        if plate_type == "long":
                            if idx != 2: force_number = True
                        elif plate_type == "square":
                            if not is_row1: force_number = True
                            elif is_row1 and idx != 2: force_number = True

                        if force_number:
                            mapping_fix = {
                                'T': '1', 'I': '1', 'J': '1', 'L': '1',
                                'Z': '2', 'S': '5', 'B': '8',
                                'D': '0', 'O': '0', 'Q': '0',
                                'A': '4', 'G': '6'
                            }
                            if char in mapping_fix: char = mapping_fix[char]

                        local_text += char
                        cv2.rectangle(self.result_debug_img, (x, y + offset_y), (x+w, y+h+offset_y), (0, 255, 0), 2)
                        cv2.putText(self.result_debug_img, char, (x, y + offset_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        idx += 1
        return local_text