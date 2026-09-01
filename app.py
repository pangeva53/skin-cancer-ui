import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import gdown
from PIL import Image
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input as resnet_preprocess

# 1. Page Configuration
st.set_page_config(
    page_title="Dermatological Lesion Multi-Model Evaluation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom Styling (Mint Theme, Matching Height/Size Cards)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #E8F5E9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #1F2937 !important;
    }

    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1350px;
    }

    /* Top Diagnosis & Confidence Cards */
    .result-card {
        height: 64px;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }

    .diag-benign {
        background-color: #DCFCE7;
        border: 1px solid #86EFAC;
        color: #15803D;
        font-weight: 700;
        font-size: 0.88rem;
    }

    .diag-malignant {
        background-color: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #B91C1C;
        font-weight: 700;
        font-size: 0.88rem;
    }

    .conf-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
    }

    .conf-title {
        font-size: 0.72rem;
        color: #6B7280;
        font-weight: 500;
        margin-bottom: 1px;
    }

    .conf-value {
        font-size: 1.15rem;
        color: #111827;
        font-weight: 700;
        line-height: 1.1;
    }

    /* Bottom Validation Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 0.4rem 0.5rem !important;
        border-radius: 6px !important;
        border: 1px solid #E5E7EB !important;
        min-height: 64px !important;
        height: auto !important;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important;
        color: #4B5563 !important;
        margin-bottom: 0px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    div[data-testid="stMetricValue"] div {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        line-height: 1.2 !important;
    }

    .stButton>button {
        width: 100% !important;
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        height: 3.2em !important;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #0369A1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load All 3 Live Models (Auto-downloads from Google Drive)
@st.cache_resource
def load_all_three_models():
    # Model 1: ResNet101 Gaussian (ROS)
    m1_file = 'model1_gauss_ros.keras'
    if not os.path.exists(m1_file):
        id_1 = '1vVTNJGIOdfUBVoR5dG-56BnG0Oug5GBG'
        gdown.download(f'https://drive.google.com/uc?id=1vVTNJGIOdfUBVoR5dG-56BnG0Oug5GBG', m1_file, quiet=False)
    m1 = tf.keras.models.load_model(m1_file)

    # Model 2: ResNet101 Raw (ROS)
    m2_file = 'model2_raw_ros.keras'
    if not os.path.exists(m2_file):
        id_2 = 'PASTE_MODEL_2_GOOGLE_DRIVE_FILE_ID_HERE'
        gdown.download(f'https://drive.google.com/uc?id=1gY9JDDnE_O8FIidKfVUOT06DlqPLMkO4', m2_file, quiet=False)
    m2 = tf.keras.models.load_model(m2_file)

    # Model 3: Head Model (SMOTE on Frozen Features)
    m3_file = 'model3_head_smote.keras'
    if not os.path.exists(m3_file):
        id_3 = 'PASTE_MODEL_3_GOOGLE_DRIVE_FILE_ID_HERE'
        gdown.download(f'https://drive.google.com/uc?id=15KYdQfHJ-M-uBcIXVWsOEAEGSkzUELFW', m3_file, quiet=False)
    m3_head = tf.keras.models.load_model(m3_file)

    # Base feature extractor for Model 3
    base_extractor = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    base_extractor.trainable = False

    return m1, m2, m3_head, base_extractor

with st.spinner("Initializing Deep Learning Models into Memory..."):
    model1_gauss, model2_raw, model3_head, base_extractor = load_all_three_models()

# 4. Preprocessing Functions
def process_gaussian(pil_img):
    img = pil_img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # 5x5 Gaussian Kernel (Sigma = 1.0)
    kernel_5x5 = tf.constant([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ], dtype=tf.float32) / 256.0
    kernel_5x5 = kernel_5x5[:, :, tf.newaxis, tf.newaxis]
    kernel_5x5 = tf.tile(kernel_5x5, [1, 1, 3, 1])
    
    blurred = tf.nn.depthwise_conv2d(img_tensor[tf.newaxis, ...], kernel_5x5, strides=[1,1,1,1], padding='SAME')
    blurred_np = tf.squeeze(blurred).numpy()
    preprocessed = resnet_preprocess(blurred_np)
    return np.expand_dims(preprocessed, axis=0)

def process_raw(pil_img):
    img = pil_img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    preprocessed = resnet_preprocess(img_array)
    return np.expand_dims(preprocessed, axis=0)

# 5. Header Section
st.title("Dermatological Lesion Multi-Model Evaluation System")
st.caption("Parallel Inference Dashboard across ResNet101 Pipelines")
st.markdown("---")

# 6. Upload & Input Display
upload_col, preview_col = st.columns([1, 1], gap="large")

with upload_col:
    st.subheader("Image Input")
    uploaded_file = st.file_uploader(
        "Upload Dermoscopic Image (Supported: JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        run_btn = st.button("Execute Multi-Model Inference")
    else:
        st.info("Upload a dermoscopic image to evaluate predictions across all three pipelines.")
        run_btn = False

with preview_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Current Case Image", width=224)

st.markdown("---")

# 7. Diagnostic Output Grid (All 3 Live Models)
if uploaded_file is not None and run_btn:
    with st.spinner("Processing image and executing inference across all 3 deep learning models..."):
        
        # Preprocess Image Arrays
        img_gauss_tensor = process_gaussian(image)
        img_raw_tensor = process_raw(image)
        
        # Model 1 Live Prediction: ResNet101 Gaussian (ROS)
        pred_raw_m1 = float(model1_gauss.predict(img_gauss_tensor, verbose=0)[0][0])
        malig_p1 = pred_raw_m1 * 100
        benign_p1 = (1.0 - pred_raw_m1) * 100
        
        # Model 2 Live Prediction: ResNet101 Raw (ROS)
        pred_raw_m2 = float(model2_raw.predict(img_raw_tensor, verbose=0)[0][0])
        malig_p2 = pred_raw_m2 * 100
        benign_p2 = (1.0 - pred_raw_m2) * 100
        
        # Model 3 Live Prediction: ResNet101 Gaussian Frozen Base (SMOTE-Tomek)
        features_m3 = base_extractor.predict(img_gauss_tensor, verbose=0)
        pred_raw_m3 = float(model3_head.predict(features_m3, verbose=0)[0][0])
        malig_p3 = pred_raw_m3 * 100
        benign_p3 = (1.0 - pred_raw_m3) * 100

        st.subheader("Comparative Diagnostic Output")
        col1, col2, col3 = st.columns(3, gap="medium")
        
        # --- MODEL 1 OUTPUT ---
        with col1:
            st.markdown("### Model 1")
            st.markdown("**ResNet101 Gaussian + ROS**")
            st.caption("Mode: **Live Model** | Sampling: ROS")
            
            d_col, c_col = st.columns([1, 1])
            with d_col:
                if pred_raw_m1 > 0.5:
                    conf1 = malig_p1
                    st.markdown('<div class="result-card diag-malignant">Diagnosis:<br>MALIGNANT</div>', unsafe_allow_html=True)
                else:
                    conf1 = benign_p1
                    st.markdown('<div class="result-card diag-benign">Diagnosis:<br>BENIGN</div>', unsafe_allow_html=True)
            with c_col:
                st.markdown(f'<div class="result-card conf-card"><div class="conf-title">Confidence</div><div class="conf-value">{conf1:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.write("")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", "86.97%")
            with m2:
                st.metric("Sensitivity", "86.97%")
            with m3:
                st.metric("Specificity", "92.10%")
            
            st.write("---")
            st.write("**Probability Breakdown**")
            st.write(f"Benign: {benign_p1:.1f}%")
            st.progress(int(np.clip(benign_p1, 0, 100)))
            st.write(f"Malignant: {malig_p1:.1f}%")
            st.progress(int(np.clip(malig_p1, 0, 100)))

        # --- MODEL 2 OUTPUT ---
        with col2:
            st.markdown("### Model 2")
            st.markdown("**ResNet101 Raw + ROS**")
            st.caption("Mode: **Live Model** | Sampling: ROS")
            
            d_col, c_col = st.columns([1, 1])
            with d_col:
                if pred_raw_m2 > 0.5:
                    conf2 = malig_p2
                    st.markdown('<div class="result-card diag-malignant">Diagnosis:<br>MALIGNANT</div>', unsafe_allow_html=True)
                else:
                    conf2 = benign_p2
                    st.markdown('<div class="result-card diag-benign">Diagnosis:<br>BENIGN</div>', unsafe_allow_html=True)
            with c_col:
                st.markdown(f'<div class="result-card conf-card"><div class="conf-title">Confidence</div><div class="conf-value">{conf2:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.write("")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", "86.67%")
            with m2:
                st.metric("Sensitivity", "86.67%")
            with m3:
                st.metric("Specificity", "91.80%")
            
            st.write("---")
            st.write("**Probability Breakdown**")
            st.write(f"Benign: {benign_p2:.1f}%")
            st.progress(int(np.clip(benign_p2, 0, 100)))
            st.write(f"Malignant: {malig_p2:.1f}%")
            st.progress(int(np.clip(malig_p2, 0, 100)))

        # --- MODEL 3 OUTPUT ---
        with col3:
            st.markdown("### Model 3")
            st.markdown("**ResNet101 Gaussian (Frozen) + SMOTE**")
            st.caption("Mode: **Live Model** | Sampling: SMOTE-Tomek")
            
            d_col, c_col = st.columns([1, 1])
            with d_col:
                if pred_raw_m3 > 0.5:
                    conf3 = malig_p3
                    st.markdown('<div class="result-card diag-malignant">Diagnosis:<br>MALIGNANT</div>', unsafe_allow_html=True)
                else:
                    conf3 = benign_p3
                    st.markdown('<div class="result-card diag-benign">Diagnosis:<br>BENIGN</div>', unsafe_allow_html=True)
            with c_col:
                st.markdown(f'<div class="result-card conf-card"><div class="conf-title">Confidence</div><div class="conf-value">{conf3:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.write("")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", "86.52%")
            with m2:
                st.metric("Sensitivity", "86.52%")
            with m3:
                st.metric("Specificity", "91.50%")
            
            st.write("---")
            st.write("**Probability Breakdown**")
            st.write(f"Benign: {benign_p3:.1f}%")
            st.progress(int(np.clip(benign_p3, 0, 100)))
            st.write(f"Malignant: {malig_p3:.1f}%")
            st.progress(int(np.clip(malig_p3, 0, 100)))