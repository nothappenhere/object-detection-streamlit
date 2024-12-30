from dotenv import load_dotenv
import streamlit as st
import numpy as np
import requests
import cv2
from roboflow import get_model_endpoint, detect_parking_space
from yolo import detect_with_local_model

load_dotenv()

st.set_page_config(
    page_title="Parking Lot Detection • Streamlit",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://github.com/nothappenhere",
        "About": "## Made with :heart: by [Muhammad Rizky Akbar](https://linkedin.com/in/mhmmdrzkyakbr).",
    },
)


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


# Judul Aplikasi
st.title("Parking Lot Detection")
st.write(
    "You can use the Pre-trained model available in [Roboflow](https://universe.roboflow.com/) by entering the **Project Id**, **Version Number**, and **API KEY**. Or use the locally available YOLOv8 model."
)

# Pilihan Model
model_source = st.radio(
    "Select Model Source:", ["Roboflow API", "Local YOLOv8"], horizontal=True
)

if model_source == "Roboflow API":
    # Input API Configuration
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        model = st.text_input(
            "Project ID",
            placeholder="e.g., parking-space",
            help="See how to get your project ID and version number [here](https://docs.roboflow.com/api-reference/workspace-and-project-ids#how-to-retrieve-a-project-id-and-version-number).",
            label_visibility="visible",
        )
    with col2:
        version = st.text_input(
            "Version Number",
            placeholder="e.g., 1",
            help="See how to get your project ID and version Number [here](https://docs.roboflow.com/api-reference/workspace-and-project-ids#how-to-retrieve-a-project-id-and-version-number).",
            label_visibility="visible",
        )
    with col3:
        API_KEY = st.text_input(
            "API KEY",
            placeholder="Insert API KEY",
            help="See how to get your API KEY [here](https://docs.roboflow.com/api-reference/authentication).",
            label_visibility="visible",
        )

st.divider()

# Pilih Input Gambar atau URL
st.subheader("Upload Method")
upload_option = st.radio("Choose input type:", ["Image", "URL"], horizontal=True)
if upload_option == "Image":
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
elif upload_option == "URL":
    image_url = st.text_input(
        "Enter Image URL",
        placeholder="e.g., https://example.com/image.jpg",
    )

st.divider()

if model_source == "Roboflow API":
    # Input Filter Tambahan
    st.subheader("Detection Filters")
    filter_col1, filter_col2, filter_col3 = st.columns([7, 1.5, 1.5])
    with filter_col1:
        filter_classes = st.text_input(
            "Filter Classes (Separate names with commas)",
            placeholder="e.g., empty, occupied",
        )
        stroke_width = st.radio(
            "Stroke Width:", ["1px", "2px", "5px", "10px"], horizontal=True
        )
        thickness = int(stroke_width.replace("px", ""))
    with filter_col2:
        min_confidence = st.number_input(
            "Min. Confidence (%)", min_value=0, max_value=100, value=25
        )
    with filter_col3:
        max_overlap = st.number_input(
            "Max. Overlap (%)", min_value=0, max_value=100, value=30
        )
elif model_source == "Local YOLOv8":
    # Input Filter Tambahan
    st.subheader("Detection Filters")
    filter_col1, filter_col2 = st.columns([2, 8])
    with filter_col1:
        min_confidence = st.number_input(
            "Min. Confidence (%)", min_value=0, max_value=100, value=25
        )

    st.divider()

# Tombol Deteksi
if st.button("Run Detection", type="primary"):
    if model_source == "Roboflow API":
        # Konfigurasi Endpoint
        model_endpoint = get_model_endpoint(API_KEY, model, version)

    if upload_option == "Image" and uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    elif upload_option == "URL" and image_url:
        response = requests.get(image_url)
        if response.status_code == 200:
            file_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
        else:
            st.error("Failed to fetch the image from URL.")
            img = None
    else:
        st.error("Please provide an image file or URL.")
        img = None

    # Proses Deteksi
    if img is not None:
        st.image(
            img, caption="Original Image", channels="BGR", use_container_width=True
        )
        with st.spinner("Running detection..."):
            if model_source == "Roboflow API":
                results = detect_parking_space(
                    img, model_endpoint, filter_classes, min_confidence, max_overlap
                )
            elif model_source == "Local YOLOv8":
                detected_image_path = detect_with_local_model(img, min_confidence)

        if model_source == "Roboflow API":
            if results:
                # Gambar dengan bounding boxes
                image_with_boxes, counter = draw_boxes_and_count(
                    img, results, filter_classes, min_confidence, thickness
                )

                # Menampilkan gambar hasil deteksi
                st.image(
                    image_with_boxes,
                    caption="Detected Parking Spaces",
                    channels="BGR",
                    use_container_width=True,
                )

                # Menampilkan jumlah ruang kosong dan terisi
                st.write(f"**Empty Spaces:** {counter.get('empty', 0)}")
                st.write(f"**Occupied Spaces:** {counter.get('occupied', 0)}")

                # Menampilkan hasil deteksi dalam format JSON
                st.json(results)
        elif model_source == "Local YOLOv8":
            if detected_image_path:
                # Menampilkan gambar hasil deteksi dari path yang disimpan
                detected_img = cv2.imread(detected_image_path)
                st.image(
                    detected_img,
                    caption="Detected Parking Spaces",
                    channels="BGR",
                    use_container_width=True,
                )
