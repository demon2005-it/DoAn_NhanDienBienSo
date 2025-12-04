# app.py
import gradio as gr
from core import ANPRSystem

# Khởi tạo hệ thống (Chỉ load model 1 lần duy nhất ở đây)
anpr_system = ANPRSystem()

def run_prediction(image):
    return anpr_system.predict(image)

if __name__ == "__main__":
    demo = gr.Interface(
        fn=run_prediction,
        inputs=gr.Image(label="Ảnh đầu vào"),
        outputs=[gr.Image(label="Ảnh đã xử lý"), gr.Textbox(label="Kết quả")],
        title="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE (ANPR)",
        description="Demo nhận diện biển số xe máy và ô tô - Đồ án IT Năm 3"
    )
    demo.launch()