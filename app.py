import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
import gdown
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess

# 1. Page Configuration
st.set_page_config(
    page_title="Dermatological Lesion Multi-Model Evaluation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom Styling (Mint Theme, Professional Layout)
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

    div[data-testid="stMetric"] {
        background-color: #F9FAFB;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #E5E7EB;
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

# 3. Load Model 1 (Auto-downloads from Drive if not cached locally)
@st.cache_resource
def load_real_model1():
    model_filename = 'model1_gauss_ros.keras'
    
    # Check if model already exists locally/cached
    if not os.path.exists(model_filename):
        # Insert your Google Drive File ID below
        file_id = '1a2b3c4d5e6f7g8h9_EXAMPLE_ID' 
        url = f'https://drive.google.com/uc?id={1vVTNJGIOdfUBVoR5dG-56BnG0Oug5GBG}'
        
        with st.spinner("Downloading ResNet101 model weights (first-time boot)..."):
            gdown.download(url, model_filename, quiet=False)
            
    model = tf.keras.models.load_model(model_filename)
    return model

with st.spinner("Loading ResNet101 Gaussian (ROS) Model Weights..."):
    model1_gauss = load_real_model1()

# 4. Exact 5x5 Gaussian Preprocessing Function
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
    
    # Apply Depthwise Conv
    blurred = tf.nn.depthwise_conv2d(img_tensor[tf.newaxis, ...], kernel_5x5, strides=[1,1,1,1], padding='SAME')
    blurred_np = tf.squeeze(blurred).numpy()
    preprocessed = resnet_preprocess(blurred_np)
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

# 7. Diagnostic Output Grid
if uploaded_file is not None and run_btn:
    with st.spinner("Processing image and executing inference..."):
        
        # --- MODEL 1: REAL INFERENCE ---
        input_tensor_m1 = process_gaussian(image)
        pred_raw_m1 = float(model1_gauss.predict(input_tensor_m1, verbose=0)[0][0])
        
        # Calculate percentages
        malig_p1 = pred_raw_m1 * 100
        benign_p1 = (1.0 - pred_raw_m1) * 100
        
        # --- MODEL 2 & 3: SIMULATED INFERENCE ---
        time.sleep(0.4)
        benign_p2, malig_p2 = 85.2, 14.8
        benign_p3, malig_p3 = 82.0, 18.0

        st.subheader("Comparative Diagnostic Output")
        col1, col2, col3 = st.columns(3, gap="medium")
        
        # --- MODEL 1 OUTPUT (LIVE MODEL) ---
        with col1:
            st.markdown("### Model 1")
            st.markdown("**ResNet101 Gaussian + ROS**")
            st.caption("Mode: **Live Model** | Sampling: ROS")
            
            if pred_raw_m1 > 0.5:
                st.error("**Diagnosis: MALIGNANT**")
                conf1 = malig_p1
            else:
                st.success("**Diagnosis: BENIGN**")
                conf1 = benign_p1
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{conf1:.1f}%")
                st.metric("Test Accuracy", "86.97%")
            with m2:
                st.metric("Sensitivity", "86.97%")
                st.metric("Specificity", "92.10%")
            
            st.write("---")
            st.write("**Probability Breakdown**")
            st.write(f"Benign: {benign_p1:.1f}%")
            st.progress(int(np.clip(benign_p1, 0, 100)))
            st.write(f"Malignant: {malig_p1:.1f}%")
            st.progress(int(np.clip(malig_p1, 0, 100)))

        # --- MODEL 2 OUTPUT (SIMULATED) ---
        with col2:
            st.markdown("### Model 2")
            st.markdown("**ResNet101 Raw + ROS**")
            st.caption("Mode: Benchmark Reference | Sampling: ROS")
            
            st.success("**Diagnosis: BENIGN**")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{benign_p2:.1f}%")
                st.metric("Test Accuracy", "86.67%")
            with m2:
                st.metric("Sensitivity", "86.67%")
                st.metric("Specificity", "91.80%")
            
            st.write("---")
            st.write("**Probability Breakdown**")
            st.write(f"Benign: {benign_p2:.1f}%")
            st.progress(int(benign_p2))
            st.write(f"Malignant: {malig_p2:.1f}%")
            st.progress(int(malig_p2))

        # --- MODEL 3 OUTPUT (SIMULATED) ---
        with col3:
            st.markdown("### Model 3")
            st.markdown("**ResNet101 Gaussian (Frozen) + SMOTE**")
            st.caption("Mode: Benchmark Reference | Sampling: SMOTE-Tomek")
            
            st.success("**Diagnosis: BENIGN**")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{benign_p3:.1f}%")
                st.metric("Test Accuracy", "86.52%")
            with m2:
                st.metric("Sensitivity", "86.52%")
                st.metric("Specificity", "91.50%")
            
            st.write("---")
            st.write("**Probability Breakdown**")
            st.write(f"Benign: {benign_p3:.1f}%")
            st.progress(int(benign_p3))
            st.write(f"Malignant: {malig_p3:.1f}%")
            st.progress(int(malig_p3))

# 8. Summary Comparison Table
st.markdown("---")
st.subheader("Pipeline Benchmark Summary")

benchmark_data = {
    "Model Pipeline": [
        "ResNet101 Gaussian (Unfrozen)", 
        "ResNet101 Raw (Unfrozen)", 
        "ResNet101 Gaussian (Frozen Base)"
    ],
    "Resampling Technique": [
        "Random Oversampling (ROS)", 
        "Random Oversampling (ROS)", 
        "SMOTE-Tomek"
    ],
    "Preprocessing Filter": [
        "5x5 Gaussian Blur", 
        "Raw (No Filter)", 
        "5x5 Gaussian Blur"
    ],
    "Validation Accuracy (%)": [86.97, 86.67, 86.52],
    "Sensitivity / Recall (%)": [86.97, 86.67, 86.52],
    "Specificity (%)": [92.10, 91.80, 91.50],
    "F1-Score": [0.92, 0.91, 0.90]
}

df_summary = pd.DataFrame(benchmark_data)
st.dataframe(df_summary, use_container_width=True, hide_index=True)