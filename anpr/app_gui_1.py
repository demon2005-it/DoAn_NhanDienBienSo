import cv2
import imutils
import numpy as np
import os
import json
import gradio as gr
from tensorflow.keras.models import load_model

# ==========================================
# 1. CẤU HÌNH & LOAD MODEL
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'anpr_cnn_model.h5')
LABEL_PATH = os.path.join(BASE_DIR, 'label_chars.json')

IMG_WIDTH = 28
IMG_HEIGHT = 28

print(f"[INFO] Đang tải model từ: {MODEL_PATH}")
model = None
reverse_label_mapping = {}

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH):
        model = load_model(MODEL_PATH)
        with open(LABEL_PATH, 'r') as f:
            label_mapping = json.load(f)
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        print("[INFO] Tải model thành công!")
    else:
        print("[CẢNH BÁO] Không tìm thấy file model hoặc json. Hãy kiểm tra lại đường dẫn.")
except Exception as e:
    print(f"[LỖI] {str(e)}")

# ==========================================
# 2. CÁC HÀM XỬ LÝ ẢNH BỔ TRỢ
# ==========================================
def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_split_line(binary_image_inv):
    horizontal_projection = np.sum(binary_image_inv, axis=1) / 255
    h = len(horizontal_projection)
    start_search = int(h * 0.2)
    end_search = int(h * 0.8)
    projection_middle = horizontal_projection[start_search:end_search]
    if projection_middle.size == 0: return int(h / 2)
    min_val = np.min(projection_middle)
    min_indices = np.where(projection_middle == min_val)[0]
    if min_indices.size > 0: min_idx_relative = min_indices[len(min_indices) // 2]
    else: min_idx_relative = (start_search + end_search) // 2
    return start_search + min_idx_relative

def sort_contours_left_to_right(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
        key=lambda b:b[1][0], reverse=False))
    return (contours, bounding_boxes)

def preprocess_char(roi):
    padding = 3 
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
        roi = roi.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        roi = roi.astype('float32') / 255.0
        return roi
    return None

# ==========================================
# 3. LOGIC CHÍNH (ĐÃ FIX LỖI TRÀN VIỀN)
# ==========================================
def predict_license_plate(input_image):
    if model is None:
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

    # --- Bước 2: Cắt biển số (Warp Perspective) ---
    pts = order_points(screenCnt)
    src_pts = pts * ratio 
    dst_w, dst_h = (450, 150) if plate_type == "long" else (280, 200)
    
    dst_pts = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(original_image, M, (dst_w, dst_h))
    
    # Vẽ contour lên ảnh gốc để hiển thị vị trí tìm thấy
    cv2.drawContours(original_image, [screenCnt.astype("int") * int(ratio)], -1, (0, 255, 0), 3)

    # --- Bước 3: Xử lý ảnh nhị phân & Làm sạch nhiễu viền ---
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    (_, binary) = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # [FIX 1] Xóa viền xung quanh ảnh nhị phân để loại bỏ khung biển số/ốc vít
    h_bin, w_bin = binary.shape
    margin = 4  # Số pixel xóa ở mỗi cạnh
    binary[0:margin, :] = 0            # Xóa mép trên
    binary[h_bin-margin:, :] = 0       # Xóa mép dưới
    binary[:, 0:margin] = 0            # Xóa mép trái
    binary[:, w_bin-margin:] = 0       # Xóa mép phải
    
    result_debug_img = warped.copy() 

    # Hàm xử lý từng vùng (dòng 1 hoặc dòng 2 hoặc cả biển dài)
    def process_contours(img_binary, offset_y=0, is_row1=False):
        local_text = ""
        cnts_char, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        roi_h, roi_w = img_binary.shape # Kích thước vùng đang xét

        if len(cnts_char) > 0:
            cnts_sorted, _ = sort_contours_left_to_right(cnts_char)
            idx = 0
            for c in cnts_sorted:
                (x, y, w, h) = cv2.boundingRect(c)
                
                # [FIX 2] Điều kiện lọc contour chặt chẽ hơn
                # Tính tỷ lệ chiều cao của contour so với chiều cao vùng ảnh
                height_ratio = h / float(roi_h)
                
                # Giữ lại nếu: 
                # - Chiều cao >= 20px
                # - Tỷ lệ w/h < 2.0 (chữ không quá bẹt)
                # - height_ratio < 0.95 (tránh nhận diện vạch kẻ dọc sát mép làm chữ I/1)
                keep_contour = False
                if (h >= 20) and (w / h < 2.0) and (0.3 < height_ratio < 0.95):
                    keep_contour = True
                
                # Loại bỏ contour dính sát lề phải (thường là rác)
                if x + w > roi_w - 3: 
                    keep_contour = False

                if keep_contour: 
                    roi = img_binary[y:y+h, x:x+w]
                    roi_cnn = preprocess_char(roi)
                    
                    if roi_cnn is not None:
                        pred = model.predict(roi_cnn, verbose=0)
                        char = reverse_label_mapping.get(np.argmax(pred), '?')
                        
                        # === LOGIC FORCE NUMBER (ÉP KIỂU SỐ/CHỮ DỰA TRÊN VỊ TRÍ) ===
                        force_number = False
                        
                        if plate_type == "long":
                            # Biển dài: 51A-12345 => Chỉ vị trí index 2 là chữ
                            if idx != 2: force_number = True
                        elif plate_type == "square":
                            # Biển vuông dòng 2: 123.45 => Toàn bộ là số
                            if not is_row1: force_number = True
                            # Biển vuông dòng 1: 29-H1 => Chỉ vị trí index 2 là chữ
                            elif is_row1 and idx != 2: force_number = True

                        # Sửa các lỗi nhận diện phổ biến
                        if force_number:
                            mapping_fix = {
                                'T': '1', 'I': '1', 'J': '1', 'L': '1',
                                'Z': '2', 'S': '5', 'B': '8',
                                'D': '0', 'O': '0', 'Q': '0',
                                'A': '4', 'G': '6'
                            }
                            if char in mapping_fix: char = mapping_fix[char]

                        local_text += char
                        cv2.rectangle(result_debug_img, (x, y + offset_y), (x+w, y+h+offset_y), (0, 255, 0), 2)
                        cv2.putText(result_debug_img, char, (x, y + offset_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        idx += 1
        return local_text

    if plate_type == "square":
        split_y = find_split_line(binary)
        # Vẽ đường phân cách để debug
        cv2.line(result_debug_img, (0, split_y), (dst_w, split_y), (0, 255, 255), 2)
        
        # [FIX 3] Thêm buffer (+/- 3px) khi cắt dòng để tránh dính pixel thừa
        t1 = process_contours(binary[0:split_y-3, :], offset_y=0, is_row1=True)
        t2 = process_contours(binary[split_y+3:, :], offset_y=split_y+3, is_row1=False)
        text_result = f"{t1} - {t2}"
    else: 
        text_result = process_contours(binary, offset_y=0, is_row1=True)

    final_display = cv2.cvtColor(result_debug_img, cv2.COLOR_BGR2RGB)
    return final_display, text_result

# ==========================================
# 4. GIAO DIỆN GRADIO
# ==========================================
if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict_license_plate,
        inputs=gr.Image(label="Ảnh đầu vào"),
        outputs=[gr.Image(label="Ảnh đã xử lý"), gr.Textbox(label="Kết quả")],
        title="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE (ANPR)",
        description="Demo nhận diện biển số xe máy và ô tô "
    )
    demo.launch()