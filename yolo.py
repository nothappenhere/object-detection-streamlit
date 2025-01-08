import os
import cv2
from ultralytics import YOLO

# Fungsi untuk deteksi menggunakan model lokal YOLOv8
def detect_with_local_model(image, confidence):
    # Load model
    stored_model = os.getenv("YOLO_MODEL")
    model = YOLO(stored_model)

    # Prediksi dengan model YOLOv8
    confidence = confidence / 100
    results = model.predict(source=image, conf=confidence)

    # Menentukan direktori tujuan untuk menyimpan gambar hasil deteksi
    base_dir = "./runs"
    base_predict_dir = "predict1"  # Awalnya di predict1
    output_dir = os.path.join(base_dir, base_predict_dir)

    # Mengecek apakah direktori sudah ada, jika ada buat direktori baru dengan penomoran
    counter = 1
    while os.path.exists(output_dir):
        counter += 1
        output_dir = os.path.join(base_dir, f"predict{counter}")

    # Membuat direktori baru
    os.makedirs(output_dir)

    # Simpan hasil deteksi ke dalam direktori yang baru dibuat
    output_image_path = os.path.join(output_dir, "detected_image.jpg")

    # Menghitung ruang kosong dan terisi
    parking_counter = {"empty": 0, "occupied": 0}
    image_with_boxes = None

    for result in results:
        image_with_boxes = result.plot()  # Gambar dengan bounding boxes
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            label = model.names[int(class_id)]  # Mendapatkan nama kelas
            if label in parking_counter:
                parking_counter[label] += 1

    # Menyimpan gambar dengan bounding boxes
    if image_with_boxes is not None:
        cv2.imwrite(output_image_path, image_with_boxes)

    return output_image_path, parking_counter
