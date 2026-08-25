import streamlit as st
import pandas as pd
import time
from PIL import Image

# 1. Page Configuration (Wide layout for optimal desktop screen utilization)
st.set_page_config(
    page_title="Dermatological Lesion Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Styling (Light Mint Theme & Professional Typography)
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
        max-width: 1300px;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #D1D5DB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Primary Action Button */
    .stButton>button {
        width: 100% !important;
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        height: 3em !important;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #0369A1 !important;
    }

    /* Table Typography */
    table {
        font-size: 0.9rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
with st.sidebar:
    st.title("Model Configuration")
    st.markdown("Select deployment parameters for diagnostic evaluation.")
    
    selected_architecture = st.selectbox(
        "Model Backbone",
        ["ResNet101", "DenseNet121", "VGG16", "MobileNetV2"],
        index=0
    )
    
    selected_sampling = st.selectbox(
        "Sampling Technique",
        ["Random Oversampling (ROS)", "SMOTE-Tomek", "Baseline (No Resampling)"],
        index=0
    )
    
    selected_preprocessing = st.selectbox(
        "Preprocessing Pipeline",
        ["Gaussian Blur Filter (5x5)", "DullRazor Artifact Removal", "Raw Images"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("System Metadata")
    st.text(f"Selected Backbone: {selected_architecture}")
    st.text(f"Resampling: {selected_sampling}")
    st.text(f"Image Resolution: 224 x 224 px")

# 4. Header Section
st.title("Dermatological Lesion Classification System")
st.caption("Clinical Decision Support Dashboard for Automated Skin Lesion Analysis")
st.markdown("---")

# 5. Main Desktop Layout (Two Main Columns)
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("Lesion Input")
    uploaded_file = st.file_uploader(
        "Upload Dermoscopic Image (Supported formats: JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Case Image", use_container_width=True)
        run_btn = st.button("Execute Diagnostic Analysis")
    else:
        st.info("Awaiting image upload. Please select a valid dermatological image file to proceed.")
        run_btn = False

with right_col:
    st.subheader("Diagnostic Evaluation")
    
    if uploaded_file is not None and run_btn:
        with st.spinner(f"Running inference via {selected_architecture} pipeline..."):
            time.sleep(1.2)  # Simulates model prediction runtime
            
            # Simulated model outputs
            benign_prob = 89.2
            malignant_prob = 10.8
            
            # Primary Classification Header
            if benign_prob > 50:
                st.success("**Classification Result: BENIGN LESION**")
            else:
                st.error("**Classification Result: MALIGNANT LESION**")
            
            # Core Performance Metrics Grid
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Model Confidence", value=f"{benign_prob:.1f}%")
            with m2:
                st.metric(label="Sensitivity (Recall)", value="86.97%")
            with m3:
                st.metric(label="Specificity", value="92.10%")
                
            st.markdown("---")
            
            # Probability Distribution Bars
            st.write("**Class Probability Distribution**")
            st.write(f"Benign Probability: {benign_prob:.1f}%")
            st.progress(int(benign_prob))
            
            st.write(f"Malignant Probability: {malignant_prob:.1f}%")
            st.progress(int(malignant_prob))
            
    elif uploaded_file is not None:
        st.write("Click 'Execute Diagnostic Analysis' on the left panel to generate model predictions.")
    else:
        st.write("No active inference. The diagnostic output will render here upon upload and execution.")

# 6. Full-Width Section: Cross-Model Benchmark Comparison (Fills Desktop Space)
st.markdown("---")
st.subheader("Cross-Model Architecture Benchmark Comparison")
st.caption("Empirical validation results across standard deep learning architectures evaluated on the dataset.")

# Comparative Results Dataset
benchmark_data = {
    "Architecture": [
        "ResNet101 (Proposed)", 
        "ResNet101", 
        "DenseNet121", 
        "VGG16", 
        "MobileNetV2"
    ],
    "Preprocessing": [
        "Gaussian Filter", 
        "Raw", 
        "Gaussian Filter", 
        "Gaussian Filter", 
        "Gaussian Filter"
    ],
    "Resampling Technique": [
        "Random Oversampling (ROS)", 
        "SMOTE-Tomek", 
        "Random Oversampling (ROS)", 
        "Random Oversampling (ROS)", 
        "Random Oversampling (ROS)"
    ],
    "Accuracy (%)": [86.97, 86.52, 85.40, 82.15, 84.60],
    "Sensitivity / Recall (%)": [86.97, 86.52, 83.20, 78.40, 81.50],
    "Specificity (%)": [92.10, 91.50, 90.10, 86.30, 89.20],
    "F1-Score": [0.92, 0.90, 0.88, 0.84, 0.87]
}

df_benchmark = pd.DataFrame(benchmark_data)
st.dataframe(df_benchmark, use_container_width=True, hide_index=True)