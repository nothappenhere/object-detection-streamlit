import streamlit as st
import requests
import cv2
import os


# Fungsi konfigurasi API Roboflow
def get_model_endpoint(api_key=None, model=None, version=None):
    if api_key and model and version:
        return f"https://detect.roboflow.com/{model}/{version}?api_key={api_key}"
    else:
        stored_model = os.getenv("ROBOFLOW_MODEL")
        stored_version = os.getenv("VERSION")
        stored_api_key = os.getenv("API_KEY")
        if not stored_api_key:
            raise ValueError(
                "API Key not found in .env file and no API Key was provided."
            )
        return f"https://detect.roboflow.com/{stored_model}/{stored_version}?api_key={stored_api_key}"


# Fungsi deteksi objek dengan filter tambahan
def detect_parking_space(image, endpoint, filter_classes, min_confidence, max_overlap):
    filters = []
    if filter_classes:
        filters.append(f"classes={filter_classes.replace(' ', '')}")
    if min_confidence:
        filters.append(f"confidence={min_confidence/100}")
    if max_overlap:
        filters.append(f"overlap={max_overlap/100}")

    # Menyusun URL dengan filter tambahan
    filter_query = "&".join(filters)
    endpoint_with_filters = f"{endpoint}&{filter_query}"

    _, img_encoded = cv2.imencode(".jpg", image)
    response = requests.post(
        endpoint_with_filters, files={"file": img_encoded.tobytes()}
    )

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}")
        return None


# Fungsi menggambar bounding box dan menghitung
def draw_boxes_and_count(image, detections, filter_classes, min_confidence, thickness):
    class_colors = {"empty": (0, 255, 0), "occupied": (0, 0, 255)}

    counter = {"empty": 0, "occupied": 0}
    allowed_classes = (
        filter_classes.replace(" ", "").split(",") if filter_classes else []
    )

    for detection in detections["predictions"]:
        confidence = detection["confidence"]
        label = detection["class"]

        # Filter berdasarkan confidence dan kelas
        if confidence < min_confidence / 100 or (
            allowed_classes and label not in allowed_classes
        ):
            continue

        x, y, w, h = (
            int(detection["x"]),
            int(detection["y"]),
            int(detection["width"]),
            int(detection["height"]),
        )

        if label in counter:
            counter[label] += 1

        color = class_colors.get(label, (255, 255, 255))
        start_point = (x - w // 2, y - h // 2)
        end_point = (x + w // 2, y + h // 2)

        cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.putText(
            image,
            f"{label} ({confidence:.2f})",
            (start_point[0], start_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return image, counter
