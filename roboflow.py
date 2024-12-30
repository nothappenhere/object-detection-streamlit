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
