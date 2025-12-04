import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_model(num_classes, img_width=28, img_height=28):
    """
    Xây dựng một mô hình CNN (Mạng nơ-ron tích chập) đơn giản
    để nhận dạng 36 ký tự (A-Z, 0-9).
    """
    model = Sequential()
    
    # Input shape: (height, width, 1) - (vì ảnh xám)
    # Lớp tích chập (Convolution) đầu tiên
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1), padding='same'))
    model.add(BatchNormalization()) # Giúp huấn luyện ổn định hơn
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Lớp tích chập thứ hai
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Lớp tích chập thứ ba
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # "Duỗi" (Flatten) dữ liệu 3D thành 1D để đưa vào lớp Dense
    model.add(Flatten())
    
    # Lớp "ẩn" (Dense)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5)) # Dropout để giảm overfitting (học vẹt)
    
    # Lớp "output" (Đầu ra)
    # num_classes là số lượng ký tự (A-Z, 0-9 = 36)
    # Dùng 'softmax' để chọn ra ký tự có xác suất cao nhất
    model.add(Dense(num_classes, activation='softmax'))
    
    # Biên dịch (Compile) mô hình
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    # Tổng cộng có 36 ký tự (26 chữ A-Z + 10 số 0-9)
    # (Chúng ta sẽ bỏ qua dấu '-' vì nó không quan trọng)
    NUM_CLASSES = 36 
    
    model = build_cnn_model(NUM_CLASSES)
    
    # In tóm tắt kiến trúc mô hình
    print("="*50)
    print("ĐÃ TẠO KIẾN TRÚC MÔ HÌNH CNN:")
    print("="*50)
    model.summary()
    
    print("\nFile 'char_recognize.py' đã sẵn sàng.")
    print("BƯỚC TIẾP THEO: Cần tạo file 'train_model.py' để HUẤN LUYỆN mô hình này.")