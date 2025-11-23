import cv2
import imutils 
import numpy as np
# import pytesseract # Đã nghỉ hưu
import re 
import argparse 
import os 
import json
from tensorflow.keras.models import load_model

# --- CÀI ĐẶT ĐỐI SỐ ĐẦU VÀO ---
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Đường dẫn đến ảnh cần nhận diện")
args = vars(ap.parse_args())

# --- CÁC THAM SỐ CỦA CNN ---
MODEL_PATH = 'anpr_cnn_model.h5'
LABEL_PATH = 'label_chars.json'
IMG_WIDTH = 28
IMG_HEIGHT = 28

# --- CÁC HÀM XỬ LÝ ẢNH CHUYÊN SÂU ---

def order_points(pts):
    """Sắp xếp 4 điểm góc của biển số theo thứ tự chuẩn."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def read_image_unicode(image_path):
    """Đọc file ảnh có dấu/tiếng Việt trên đường dẫn."""
    try:
        if not os.path.exists(image_path):
            print(f"LỖI: Không tìm thấy file tại: {image_path}")
            return None
        img_stream = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

def find_split_line(binary_image_inv):
    """
    Kỹ thuật Phép chiếu ngang (Horizontal Projection) để tìm đường phân cách giữa 2 dòng.
    """
    horizontal_projection = np.sum(binary_image_inv, axis=1) / 255 
    
    h = len(horizontal_projection)
    start_search = int(h * 0.2)
    end_search = int(h * 0.8)
    
    projection_middle = horizontal_projection[start_search:end_search]
    
    if projection_middle.size == 0:
        return int(h / 2)
        
    min_val = np.min(projection_middle)
    min_indices = np.where(projection_middle == min_val)[0]
    
    if min_indices.size > 0:
        min_idx_relative = min_indices[len(min_indices) // 2]
    else:
        min_idx_relative = (start_search + end_search) // 2
        
    split_line_y = start_search + min_idx_relative
    
    return split_line_y

def sort_contours_left_to_right(contours):
    """Sắp xếp các contour (ký tự) từ trái sang phải."""
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
        key=lambda b:b[1][0], reverse=False))
    return (contours, bounding_boxes)

def preprocess_char_for_cnn(char_roi):
    """Chuẩn bị ảnh ROI ký tự để đưa vào CNN."""
    # Thêm một chút padding viền trắng (giống lúc train)
    padding = 5
    if char_roi.shape[0] > 0 and char_roi.shape[1] > 0:
        char_roi = cv2.copyMakeBorder(char_roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0) # 0 = Đen (nền)
    else:
        return None # Bỏ qua nếu ROI rỗng

    img = cv2.resize(char_roi, (IMG_WIDTH, IMG_HEIGHT))
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    img = img.astype('float32') / 255.0
    return img

# --- CHƯƠNG TRÌNH CHÍNH ---

# 1. TẢI "BỘ NÃO" CNN VÀ BẢN ĐỒ NHÃN
print("[INFO] Tải mô hình CNN và nhãn...")
try:
    model = load_model(MODEL_PATH)
    with open(LABEL_PATH, 'r') as f:
        label_mapping = json.load(f)
    
    # Đảo ngược bản đồ nhãn để tra cứu (ví dụ: 0 -> '0', 10 -> 'A')
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

except Exception as e:
    print(f"[LỖI] Không thể tải mô hình '{MODEL_PATH}' hoặc file nhãn '{LABEL_PATH}'.")
    print(f"Bạn đã chạy 'python train_model.py' chưa? Lỗi: {e}")
    exit()

# 2. ĐỌC ẢNH ĐẦU VÀO
image_path = args["image"] 
original_image = read_image_unicode(image_path) 
if original_image is None: exit() 

image = original_image.copy()
image = imutils.resize(image, width=600)
print(f"Đã đọc ảnh, kích thước gốc: {original_image.shape}, xử lý ở: {image.shape}")

# 3. TIỀN XỬ LÝ (TÌM CẠNH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
edged = cv2.Canny(blurred, 30, 180) 

# 4. TÌM BIỂN SỐ (LỌC CONTOURS)
screenCnt = None 
found_ratio = 0.0  
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20] 

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx) 
        if h == 0: continue 
        aspectRatio = w / float(h)
        is_long_plate = (aspectRatio > 2.8 and aspectRatio < 7.0) 
        is_square_plate = (aspectRatio > 1.0 and aspectRatio < 2.5) 
        if is_long_plate or is_square_plate:
            screenCnt = approx
            found_ratio = aspectRatio 
            break 

# 5. XỬ LÝ NẾU TÌM THẤY BIỂN SỐ
if screenCnt is not None:
    pts = order_points(screenCnt)
    ratio_scale = original_image.shape[1] / float(image.shape[1]) 
    src_pts = pts * ratio_scale

    if found_ratio > 2.8: 
        maxWidth = 450
        maxHeight = 150
        plate_type = "long"
    else: 
        maxWidth = 280
        maxHeight = 130 
        plate_type = "square"

    # NẮN THẲNG BIỂN SỐ
    dst_pts = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_plate = cv2.warpPerspective(original_image, M, (maxWidth, maxHeight))
    warped_plate = imutils.resize(warped_plate, height=100)
    
    warped_plate_color = warped_plate.copy()
    
    # 5b. XỬ LÝ ẢNH NHỊ PHÂN
    print(f"[INFO] Bắt đầu phân đoạn và nhận dạng ký tự (Loại: {plate_type})...")
    gray_plate = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)
    
    (T, binary_plate_inv) = cv2.threshold(gray_plate, 0, 255, 
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    cv2.imshow("Bien So Da Nan Thang (Nhi Phan)", binary_plate_inv)

    # 5c. PHÂN ĐOẠN, NHẬN DẠNG BẰNG CNN
    
    final_plate_text = ""
    
    if plate_type == "square":
        split_y = find_split_line(binary_plate_inv)
        
        row1_img = binary_plate_inv[0:split_y, :]
        row2_img = binary_plate_inv[split_y:, :]
        
        (cnts1, _) = cv2.findContours(row1_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts2, _) = cv2.findContours(row2_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Sắp xếp và Đọc Dòng 1
        text_row1 = ""
        if len(cnts1) > 0:
            (cnts1_sorted, boxes1) = sort_contours_left_to_right(cnts1)
            for c in cnts1_sorted:
                (x, y, w, h) = cv2.boundingRect(c)
                if h > 10 and w/h < 1.5: 
                    char_roi = row1_img[y:y+h, x:x+w]
                    char_cnn = preprocess_char_for_cnn(char_roi)
                    if char_cnn is None: continue # Bỏ qua nếu ROI rỗng
                    
                    prediction = model.predict(char_cnn, verbose=0) 
                    predicted_label_index = np.argmax(prediction)
                    predicted_char = reverse_label_mapping.get(predicted_label_index, '?') 
                    text_row1 += predicted_char
                    
                    cv2.rectangle(warped_plate_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(warped_plate_color, predicted_char, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 6. Sắp xếp và Đọc Dòng 2
        text_row2 = ""
        if len(cnts2) > 0:
            (cnts2_sorted, boxes2) = sort_contours_left_to_right(cnts2)
            for c in cnts2_sorted:
                (x, y, w, h) = cv2.boundingRect(c)
                if h > 10 and w/h < 1.5:
                    char_roi = row2_img[y:y+h, x:x+w]
                    char_cnn = preprocess_char_for_cnn(char_roi)
                    if char_cnn is None: continue # Bỏ qua nếu ROI rỗng
                    
                    prediction = model.predict(char_cnn, verbose=0) 
                    predicted_label_index = np.argmax(prediction)
                    predicted_char = reverse_label_mapping.get(predicted_label_index, '?')
                    
                    if predicted_char == 'T':
                        predicted_char = '1' 
                    
                    text_row2 += predicted_char
                    
                    cv2.rectangle(warped_plate_color, (x, y + split_y), (x+w, y + split_y + h), (0, 255, 0), 2)
                    cv2.putText(warped_plate_color, predicted_char, (x, y + split_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        final_plate_text = f"{text_row1} {text_row2}"
        print(f"[INFO] Kết quả thô CNN: {text_row1} | {text_row2}")

    ### === CODE MỚI BỔ SUNG (BẮT ĐẦU) === ###
    elif plate_type == "long":
        # Biển dài chỉ có 1 dòng, ta tìm contour trên toàn bộ ảnh nhị phân
        (cnts, _) = cv2.findContours(binary_plate_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_row = ""
        
        if len(cnts) > 0:
            # Sắp xếp từ trái sang phải
            (cnts_sorted, boxes) = sort_contours_left_to_right(cnts)
            
            for c in cnts_sorted:
                (x, y, w, h) = cv2.boundingRect(c)
                
                # Lọc nhiễu: ký tự phải đủ cao và tỷ lệ hợp lý
                aspect_ratio = w / float(h)
                
                # Biển dài thường có ký tự cao hơn biển vuông một chút
                if h > 20 and aspect_ratio < 1.5: 
                    # Tách ROI ký tự
                    char_roi = binary_plate_inv[y:y+h, x:x+w]
                    
                    # Chuẩn bị cho CNN
                    char_cnn = preprocess_char_for_cnn(char_roi)
                    if char_cnn is None: continue # Bỏ qua nếu ROI rỗng
                    
                    # Nhận dạng
                    prediction = model.predict(char_cnn, verbose=0)
                    predicted_label_index = np.argmax(prediction)
                    predicted_char = reverse_label_mapping.get(predicted_label_index, '?')
                    
                    text_row += predicted_char
                    
                    # Vẽ kết quả lên ảnh màu
                    cv2.rectangle(warped_plate_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(warped_plate_color, predicted_char, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        final_plate_text = text_row
        print(f"[INFO] Kết quả thô CNN (biển dài): {final_plate_text}")
    ### === CODE MỚI BỔ SUNG (KẾT THÚC) === ###

    # 5d. HIỂN THỊ KẾT QUẢ CUỐI CÙNG (CNN)
    print("="*50)
    print(f"KẾT QUẢ NHẬN DẠNG (CNN): {final_plate_text}")
    print("="*50)
    
    cv2.imshow("Ket Qua Phan Doan & Nhan Dang (CNN)", warped_plate_color)

else:
    print("KHÔNG TÌM THẤY HÌNH NÀO CÓ TỶ LỆ GIỐNG BIỂN SỐ.")

# 6. HIỂN THỊ KẾT QUẢ
cv2.imshow("Anh Goc (Da Tim Kiem)", image)
print("Nhấn phím bất kỳ để thoát...")
cv2.waitKey(0) 
cv2.destroyAllWindows()