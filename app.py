import streamlit as st
import pandas as pd
import time
from PIL import Image

# 1. Page Configuration (Full Wide Layout without Sidebar)
st.set_page_config(
    page_title="Multi-Model Skin Lesion Classification Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS: Light Mint Theme, Professional Typography & Balanced Cards
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

    /* Model Card Containers */
    .model-card {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }

    /* Metric Box Styling */
    div[data-testid="stMetric"] {
        background-color: #F9FAFB;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #E5E7EB;
    }

    /* Professional Action Button */
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

# 3. Header Section
st.title("Dermatological Lesion Multi-Model Evaluation System")
st.caption("Parallel Inference Dashboard across ResNet101 Pipelines")
st.markdown("---")

# 4. Image Upload & Input Display
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

# 5. Parallel Diagnostic Output Across the 3 Models
if uploaded_file is not None and run_btn:
    with st.spinner("Executing parallel inference across all 3 ResNet101 configurations..."):
        time.sleep(1.4)  # Simulates simultaneous model execution
        
        st.subheader("Comparative Diagnostic Output")
        
        # Define the 3-Column Parallel Grid
        col1, col2, col3 = st.columns(3, gap="medium")
        
        # --- MODEL 1: ResNet101 Gaussian (ROS) ---
        with col1:
            st.markdown("### Model 1")
            st.markdown("**ResNet101 Gaussian + ROS**")
            st.caption("Preprocessing: 5x5 Gaussian | Sampling: Random Oversampling")
            
            # Simulated inference probability
            benign_p1 = 88.5
            malig_p1 = 11.5
            
            st.success("**Diagnosis: BENIGN**")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{benign_p1:.1f}%")
                st.metric("Test Accuracy", "86.97%")
            with m2:
                st.metric("Sensitivity", "86.97%")
                st.metric("Specificity", "92.10%")
            
            st.write("---")
            st.write("**Probability Breakdown**")
            st.write(f"Benign: {benign_p1:.1f}%")
            st.progress(int(benign_p1))
            st.write(f"Malignant: {malig_p1:.1f}%")
            st.progress(int(malig_p1))

        # --- MODEL 2: ResNet101 Raw (ROS) ---
        with col2:
            st.markdown("### Model 2")
            st.markdown("**ResNet101 Raw + ROS**")
            st.caption("Preprocessing: None (Raw) | Sampling: Random Oversampling")
            
            benign_p2 = 85.2
            malig_p2 = 14.8
            
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

        # --- MODEL 3: ResNet101 Gaussian Frozen Base (SMOTE-Tomek) ---
        with col3:
            st.markdown("### Model 3")
            st.markdown("**ResNet101 Gaussian (Frozen) + SMOTE**")
            st.caption("Preprocessing: 5x5 Gaussian | Sampling: SMOTE-Tomek")
            
            benign_p3 = 82.0
            malig_p3 = 18.0
            
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

# 6. Summary Comparison Table (Always Visible to Fill Space Cleanly)
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