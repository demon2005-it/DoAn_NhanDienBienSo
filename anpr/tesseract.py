import cv2
import numpy as np
import pytesseract
import os
import argparse
import re

# --- CẤU HÌNH ---
# Đảm bảo đường dẫn đúng
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred

def fix_ocr_error(text):
    """
    Sửa lỗi nhận diện nhầm: O->0, I->1, B->8...
    """
    text = text.upper()
    mapping = {
        'O': '0', 'Q': '0', 'D': '0',
        'I': '1', 'L': '1', '|': '1',
        'Z': '2', 'S': '5', 'B': '8', 'G': '6'
    }
    
    chars = list(text)
    # Chỉ sửa 2 ký tự đầu (Mã vùng) nếu nó bị nhận nhầm thành chữ
    for i in range(min(2, len(chars))):
        if chars[i] in mapping:
            chars[i] = mapping[chars[i]]
            
    return "".join(chars)

def recognize_plate_super_resolution(roi):
    """
    PHIÊN BẢN NÂNG CẤP: Phóng to ảnh (Upscale) để đọc các số nhỏ, dính nhau
    """
    # 1. PHÓNG TO ẢNH (Quan trọng nhất)
    # Phóng to gấp 3 lần bằng thuật toán Cubic (giữ nét tốt hơn)
    roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 2. Chuyển xám và làm mịn nhẹ
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Blur nhẹ để giảm nhiễu hạt do phóng to
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3. Thử 2 bộ lọc nhị phân
    # Cách A: Otsu (Tốt cho ảnh sáng đều)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Cách B: Adaptive (Tốt cho ảnh bị bóng đổ/chói sáng)
    # Block size tăng lên 35 vì ảnh đã phóng to
    thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)

    candidates = [thresh_otsu, thresh_adapt]
    
    # Cấu hình Tesseract: --psm 7 (Single line) là chuẩn nhất cho biển số cắt rời
    config = "--psm 7 -c tesseract_char_whitelist=ABCDEFGHKLMNPQRSTUVWXYZ0123456789"
    
    best_text = ""
    
    for img_bin in candidates:
        # 4. THÊM VIỀN TRẮNG (Padding)
        # Tesseract rất ghét chữ dính sát mép ảnh, thêm viền trắng bao quanh giúp nó đọc tốt hơn
        img_bin = cv2.copyMakeBorder(img_bin, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # Debug: Xem ảnh phóng to trông thế nào
        # cv2.imshow("Anh Super Resolution", img_bin)
        
        raw_text = pytesseract.image_to_string(img_bin, lang='eng', config=config)
        
        # Lọc bỏ ký tự lạ
        clean_text = "".join(c for c in raw_text if c.isalnum())
        fixed_text = fix_ocr_error(clean_text)
        
        # Logic chọn kết quả tốt nhất:
        # Ưu tiên kết quả dài hơn (vì thường bị đọc thiếu số)
        if len(fixed_text) > len(best_text):
            best_text = fixed_text
            
        # Nếu đọc được >= 7 ký tự (chuẩn biển số VN) -> Trả về luôn
        if len(fixed_text) >= 7:
            return fixed_text, img_bin

    return best_text, thresh_adapt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to image")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Khong tim thay file: {args.image}")
        return

    img = cv2.imread(args.image)
    target_width = 800
    scale = target_width / img.shape[1]
    img = cv2.resize(img, (target_width, int(img.shape[0] * scale)))
    display_img = img.copy()

    # Tìm vùng
    gray, blurred = preprocess_image(img)
    edges = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    
    found = False

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1000: continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)

        # Tỷ lệ khung hình (Lỏng một chút để bắt dính)
        if 1.5 <= aspect_ratio <= 6.0:
            # CẮT ẢNH (ROI)
            # Cắt sâu vào 5% để loại bỏ viền đen khung biển
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.10)
            
            h_img, w_img = img.shape[:2]
            y_start = max(0, y + pad_y)
            y_end = min(h_img, y + h - pad_y)
            x_start = max(0, x + pad_x)
            x_end = min(w_img, x + w - pad_x)
            
            roi = img[y_start:y_end, x_start:x_end]
            
            if roi.size == 0: continue

            # GỌI HÀM NHẬN DIỆN NÂNG CẤP
            text, bin_img = recognize_plate_super_resolution(roi)
            
            if text:
                print(f" > Thu Contour {i}: {text}")
                
                # KIỂM TRA QUY TẮC:
                # 1. Độ dài >= 6 (Ví dụ: 29A1234)
                # 2. Bắt đầu bằng số (Mã vùng)
                if len(text) >= 6 and re.match(r"^[0-9]", text):
                    print(f"✅ KET QUA: {text}")
                    
                    # Vẽ kết quả
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    # Vẽ chữ to, đậm để dễ nhìn
                    cv2.putText(display_img, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Hiển thị ảnh nhị phân đã phóng to để kiểm tra
                    cv2.imshow("Binary OCR (Zoomed)", bin_img)
                    found = True
                    break

    if not found:
        print("❌ Khong doc duoc ro rang.")
    
    cv2.imshow("Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()