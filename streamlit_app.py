import streamlit as st
import os
import tempfile
from detect_anomaly import run_anomaly_detection

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="CCTV Anomaly Detection",
    layout="wide",
    page_icon="🚨"
)

# =========================
# Background Styling
# =========================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1549923746-c502d488b3ea?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.block-container {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 12px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# Fake Login / Signup
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 CCTV Anomaly Detection Login")
    option = st.radio("Choose Action:", ["Login", "Sign Up"])

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Submit"):
        if option == "Login":
            if username == "admin" and password == "1234":  # demo creds
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password")
        else:
            st.success("🎉 Account created successfully! Please login.")
else:
    # =========================
    # Main App
    # =========================
    st.markdown("<h1 style='text-align:center'>🚨 CCTV Anomaly Detection using ConvLSTM Autoencoder</h1>", unsafe_allow_html=True)
    st.write("Upload a CCTV video and our AI model will detect anomalies and overlay a heatmap on suspicious frames.")

    uploaded_file = st.file_uploader("📤 Upload a CCTV video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded video to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        input_path = tfile.name

        # Display videos side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎥 Original Video")
            st.video(input_path)

        st.info("⏳ Processing video... please wait while AI analyzes it.")

        output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
        run_anomaly_detection(input_path, output_path)

        if os.path.exists(output_path):
            with col2:
                st.subheader("🔍 Anomaly Detected Video")
                st.video(output_path)
            st.success("✅ Processing complete!")
        else:
            st.error("❌ No output video generated. Please check backend logs.")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()
