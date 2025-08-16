import streamlit as st
import os
import tempfile
import cv2
from detect_anomaly import run_anomaly_detection

st.set_page_config(page_title="CCTV Anomaly Detection", layout="centered")

st.title("🚨 CCTV Anomaly Detection using ConvLSTM Autoencoder")
st.write("Upload a video and the system will detect anomalies and generate a heatmap overlay.")

# File uploader
uploaded_file = st.file_uploader("Upload a CCTV video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    st.video(input_path)  # Show original video

    st.info("⏳ Processing video, please wait...")

    # Output path
    output_path = os.path.join(tempfile.gettempdir(), "output.mp4")

    # Run anomaly detection
    run_anomaly_detection(input_path, output_path)

    if os.path.exists(output_path):
        st.success("Anomaly detection complete. See the result below:")
        st.video(output_path)
    else:
        st.error(" No output video generated. Please check logs.")
