# DoAn_NhanDienBienSo
Đồ án môn Xử lý ảnh - Nhận diện biển số xe (Nhóm 3)

# 🧠 Nguyên Lý Hoạt Động (System Architecture & Logic)

Hệ thống nhận diện biển số xe (ANPR) này được xây dựng dựa trên sự kết hợp giữa **Computer Vision (OpenCV)** để xử lý ảnh và **Deep Learning (CNN)** để nhận diện ký tự.

Quy trình hoạt động được chia thành 2 giai đoạn chính:

## I. Giai đoạn 1: Huấn luyện Mô hình (Model Training)
*File liên quan: `char_recognize.py`, `train_model.py`*

Trước khi hệ thống có thể hoạt động, một mô hình CNN (Convolutional Neural Network) được huấn luyện để nhận biết các ký tự (A-Z, 0-9).

1.  **Kiến trúc mạng CNN:**
    * **Input:** Ảnh xám (Grayscale) kích thước chuẩn **28x28 pixel**.
    * **Feature Extraction (Trích xuất đặc trưng):** Sử dụng 3 lớp tích chập (`Conv2D`) kết hợp với `MaxPooling2D` để học các đường nét, góc cạnh của ký tự. Số lượng bộ lọc (filters) tăng dần: 32 -> 64 -> 128.
    * **Classification (Phân loại):** Lớp `Flatten` duỗi phẳng dữ liệu và đưa vào lớp `Dense` (Fully Connected) để đưa ra dự đoán xác suất cho từng ký tự.
2.  **Dữ liệu:**
    * Mô hình được huấn luyện trên bộ dữ liệu ký tự đã gán nhãn.
    * Sử dụng hàm mất mát `categorical_crossentropy` và tối ưu hóa bằng `Adam`.

---

## II. Giai đoạn 2: Quy trình Nhận diện (Inference Pipeline)
*File liên quan: `anpr.py`*

Khi người dùng tải ảnh hoặc video lên, hệ thống xử lý theo luồng **Pipeline 4 bước** sau:

### 📍 Bước 1: Phát hiện vị trí biển số (License Plate Detection)
Mục tiêu: Tìm ra tọa độ hình chữ nhật chứa biển số trong bức ảnh lớn.
* **Tiền xử lý:** Ảnh gốc -> Resize -> Chuyển sang ảnh xám (Grayscale) -> Làm mờ (Gaussian Blur) để giảm nhiễu -> Tách biên (Canny Edge Detection).
* **Tìm Contours:** Tìm tất cả các đường bao khép kín trong ảnh.
* **Lọc ứng viên:** Thuật toán duyệt qua các contour và lọc dựa trên hình học:
    * Hình phải có 4 góc (xấp xỉ hình chữ nhật).
    * **Tỉ lệ khung hình (Aspect Ratio):**
        * `2.5 <= Ratio <= 7.0`: Nhận diện là **Biển dài**.
        * `1.0 <= Ratio < 2.5`: Nhận diện là **Biển vuông** (2 dòng).

### 📐 Bước 2: Cắt & Biến đổi hình học (Warp Perspective)
Ảnh chụp thực tế thường bị nghiêng hoặc méo. Bước này giúp đưa biển số về góc nhìn thẳng ("Scan" ảnh).
* Sử dụng hàm `order_points` để xác định 4 góc: Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái.
* Áp dụng **Perspective Transform** (Biến đổi phối cảnh) để cắt vùng biển số và "nắn" thẳng lại thành hình chữ nhật chuẩn.

### ✂️ Bước 3: Phân đoạn ký tự (Character Segmentation)
Mục tiêu: Tách rời từng ký tự ra khỏi nền biển số.
1.  **Nhị phân hóa (Binarization):** Dùng thuật toán **Thresholding (Otsu)** để chuyển ảnh về dạng đen-trắng hoàn toàn.
2.  **Xử lý biển vuông (Đặc biệt):**
    * Nếu là biển vuông, hệ thống dùng thuật toán **Horizontal Projection** (Cộng gộp pixel theo chiều ngang) để tìm đường rãnh ngăn cách giữa dòng trên và dòng dưới (`find_split_line`).
3.  **Lọc nhiễu:**
    * Tìm contour các vùng trắng.
    * Loại bỏ các vùng nhiễu (vết bẩn, ốc vít, viền) dựa trên diện tích và tỉ lệ chiều cao/chiều rộng. Chỉ giữ lại các vùng có hình dáng giống ký tự.
4.  **Sắp xếp:** Sắp xếp các ký tự từ Trái sang Phải (và Trên xuống Dưới đối với biển vuông).

### 🤖 Bước 4: Nhận diện & Hậu xử lý Logic (Recognition & Heuristic)
Đây là bước quan trọng nhất để đảm bảo độ chính xác cao.

1.  **Dự đoán:** Từng ảnh ký tự sau khi cắt được đưa vào model CNN đã train ở Giai đoạn 1 để dự đoán.
2.  **Thuật toán Sửa lỗi Logic (Heuristic Correction):**
    Do AI có thể nhầm lẫn giữa các ký tự giống nhau (Ví dụ: `8` và `B`, `0` và `D`), hệ thống áp dụng các quy luật biển số xe Việt Nam để ép kiểu dữ liệu:
    
    * **Quy luật chung:**
        * **Biển dài:** Ký tự thứ 3 luôn là **CHỮ**, các ký tự còn lại ưu tiên **SỐ**.
        * **Biển vuông:** Dòng 1 (2 ký tự đầu là Mã tỉnh - Số, ký tự thứ 3 là Series - Chữ). Dòng 2 luôn là **SỐ**.
        
    * **Bảng ánh xạ sửa lỗi (Correction Map):**
        * Nếu vị trí đó bắt buộc là **SỐ**: Ép `Z` -> `2`, `S` -> `5`, `B` -> `8`, `D` -> `0`,...
        * Nếu vị trí đó bắt buộc là **CHỮ**: Ép `4` -> `A`, `8` -> `B`, `0` -> `D`,...

---

## 🛠 Công nghệ sử dụng
| Công nghệ | Mục đích |
| :--- | :--- |
| **Python** | Ngôn ngữ lập trình chính. |
| **OpenCV** | Xử lý ảnh (Canny, Threshold, FindContours, WarpPerspective). |
| **TensorFlow / Keras** | Xây dựng và chạy mô hình Deep Learning (CNN). |
| **NumPy** | Xử lý ma trận ảnh và tính toán hình học. |
| **Gradio** | Xây dựng giao diện Web App tương tác (UI). |
