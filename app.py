import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import gc
import gdown
from PIL import Image
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input as resnet_preprocess

import db

# 1. Page Configuration
st.set_page_config(
    page_title="Dermatological Lesion Multi-Model Evaluation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Database Schema
db.init_db()

# 2. Custom Styling
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

    /* Diagnosis & Confidence Cards */
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
    }

    div[data-testid="stMetricValue"] div {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        line-height: 1.2 !important;
    }

    .summary-box {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-left: 5px solid #0284C7;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-top: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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
    }

    .stButton>button:hover {
        background-color: #0369A1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Session State Management
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

def render_auth_page():
    st.title("Clinical Diagnostic System Authentication")
    st.caption("Secure decision-support portal for authorized personnel.")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        tab_login, tab_register = st.tabs(["Sign In", "Register Clinician Account"])
        
        with tab_login:
            st.subheader("Account Login")
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Log In"):
                if db.verify_user(login_user, login_pass):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = login_user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            st.caption("Default credentials: User: `admin` | Password: `admin123`")

        with tab_register:
            st.subheader("New Account Registration")
            reg_user = st.text_input("Choose Username", key="reg_user")
            reg_pass = st.text_input("Choose Password", type="password", key="reg_pass")
            
            if st.button("Create Account"):
                if reg_user and reg_pass:
                    success, msg = db.register_user(reg_user, reg_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please provide both a username and password.")

if not st.session_state["authenticated"]:
    render_auth_page()
    st.stop()

# 4. Header Section & Session Controls
head_left, head_right = st.columns([3, 1])
with head_left:
    st.title("Dermatological Lesion Multi-Model Evaluation System")
    st.caption("Parallel Inference Dashboard across ResNet101 Pipelines")
with head_right:
    st.write(f"Logged in as: **{st.session_state['username']}**")
    if st.button("Log Out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()

st.markdown("---")

# 5. Helper Functions (Downloads, Filters, Sequential Prediction)
def fetch_weights(filename, file_id):
    if not os.path.exists(filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)

def process_gaussian(pil_img):
    img = pil_img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
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

def execute_low_memory_inference(img_gauss, img_raw):
    # Model 1
    fetch_weights('model1_gauss_ros.keras', '1vVTNJGIOdfUBVoR5dG-56BnG0Oug5GBG')
    m1 = tf.keras.models.load_model('model1_gauss_ros.keras', compile=False)
    p1 = float(m1.predict(img_gauss, verbose=0)[0][0])
    del m1
    tf.keras.backend.clear_session()
    gc.collect()

    # Model 2
    fetch_weights('model2_raw_ros.keras', '1gY9JDDnE_O8FIidKfVUOT06DlqPLMkO4')
    m2 = tf.keras.models.load_model('model2_raw_ros.keras', compile=False)
    p2 = float(m2.predict(img_raw, verbose=0)[0][0])
    del m2
    tf.keras.backend.clear_session()
    gc.collect()

    # Model 3
    fetch_weights('model3_head_smote.keras', '15KYdQfHJ-M-uBcIXVWsOEAEGSkzUELFW')
    base_extractor = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    feats = base_extractor.predict(img_gauss, verbose=0)
    del base_extractor
    tf.keras.backend.clear_session()
    gc.collect()

    m3_head = tf.keras.models.load_model('model3_head_smote.keras', compile=False)
    p3 = float(m3_head.predict(feats, verbose=0)[0][0])
    del m3_head
    tf.keras.backend.clear_session()
    gc.collect()

    return p1, p2, p3

# 6. Tabbed Dashboard Navigation
tab_diag, tab_records = st.tabs(["Diagnostic Inference", "Patient Database Records"])

# ==================== TAB 1: DIAGNOSTIC INFERENCE ====================
with tab_diag:
    upload_col, preview_col = st.columns([1, 1], gap="large")

    with upload_col:
        st.subheader("Patient & Image Input")
        # Patient Name input field before file upload
        patient_name = st.text_input("Patient Full Name / ID Code", placeholder="e.g., John Doe (P-1002)")
        
        uploaded_file = st.file_uploader(
            "Upload Dermoscopic Image (Supported: JPG, JPEG, PNG)", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None and patient_name.strip() != "":
            run_btn = st.button("Execute Multi-Model Inference")
        else:
            if uploaded_file is not None and patient_name.strip() == "":
                st.warning("Please enter the patient's name before executing inference.")
            else:
                st.info("Enter patient details and upload a dermoscopic image.")
            run_btn = False

    with preview_col:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption=f"Case Image: {patient_name}", width=224)

    st.markdown("---")

    if uploaded_file is not None and run_btn:
        with st.spinner("Executing low-memory sequential inference across all 3 models..."):
            # Convert uploaded image to bytes for BLOB storage
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            
            img_gauss_tensor = process_gaussian(image)
            img_raw_tensor = process_raw(image)
            
            pred_raw_m1, pred_raw_m2, pred_raw_m3 = execute_low_memory_inference(img_gauss_tensor, img_raw_tensor)

            malig_p1 = pred_raw_m1 * 100
            benign_p1 = (1.0 - pred_raw_m1) * 100
            diag_m1 = "MALIGNANT" if pred_raw_m1 > 0.5 else "BENIGN"
            conf1 = malig_p1 if pred_raw_m1 > 0.5 else benign_p1

            malig_p2 = pred_raw_m2 * 100
            benign_p2 = (1.0 - pred_raw_m2) * 100
            diag_m2 = "MALIGNANT" if pred_raw_m2 > 0.5 else "BENIGN"
            conf2 = malig_p2 if pred_raw_m2 > 0.5 else benign_p2

            malig_p3 = pred_raw_m3 * 100
            benign_p3 = (1.0 - pred_raw_m3) * 100
            diag_m3 = "MALIGNANT" if pred_raw_m3 > 0.5 else "BENIGN"
            conf3 = malig_p3 if pred_raw_m3 > 0.5 else benign_p3

            # Log prediction with patient name and image bytes
            db.log_prediction(
                st.session_state["username"],
                patient_name,
                diag_m1, conf1,
                diag_m2, conf2,
                diag_m3, conf3,
                image_bytes
            )

            st.subheader("Comparative Diagnostic Output")
            col1, col2, col3 = st.columns(3, gap="medium")
            
            # --- MODEL 1 OUTPUT ---
            with col1:
                st.markdown("### Model 1")
                st.markdown("**ResNet101 Gaussian + ROS**")
                st.caption(f"Patient: **{patient_name}** | Mode: Live")
                
                d_col, c_col = st.columns([1, 1])
                with d_col:
                    if diag_m1 == "MALIGNANT":
                        st.markdown('<div class="result-card diag-malignant">Diagnosis:<br>MALIGNANT</div>', unsafe_allow_html=True)
                    else:
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
                st.caption(f"Patient: **{patient_name}** | Mode: Live")
                
                d_col, c_col = st.columns([1, 1])
                with d_col:
                    if diag_m2 == "MALIGNANT":
                        st.markdown('<div class="result-card diag-malignant">Diagnosis:<br>MALIGNANT</div>', unsafe_allow_html=True)
                    else:
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
                st.caption(f"Patient: **{patient_name}** | Mode: Live")
                
                d_col, c_col = st.columns([1, 1])
                with d_col:
                    if diag_m3 == "MALIGNANT":
                        st.markdown('<div class="result-card diag-malignant">Diagnosis:<br>MALIGNANT</div>', unsafe_allow_html=True)
                    else:
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

            # Color theme switching based on diagnosis
            badge_bg = "#FEE2E2" if diag_m1 == "MALIGNANT" else "#DCFCE7"
            badge_color = "#991B1B" if diag_m1 == "MALIGNANT" else "#166534"
            border_accent = "#EF4444" if diag_m1 == "MALIGNANT" else "#10B981"

            st.markdown(f"""
            <div style="
                background-color: #FFFFFF;
                border-radius: 10px;
                border: 1px solid #E5E7EB;
                border-left: 6px solid {border_accent};
                padding: 1.25rem 1.5rem;
                margin-top: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <h4 style="margin: 0; color: #0F172A; font-size: 1.15rem; font-weight: 700;">
                        Diagnostic Synthesis & Recommended Model Decision
                    </h4>
                    <span style="
                        background-color: {badge_bg};
                        color: {badge_color};
                        font-size: 0.8rem;
                        font-weight: 700;
                        padding: 4px 10px;
                        border-radius: 9999px;
                        text-transform: uppercase;
                    ">
                        Primary Verdict: {diag_m1}
                    </span>
                </div>

                <div style="margin-bottom: 0.75rem; line-height: 1.6; font-size: 0.92rem; color: #334155;">
                    <span style="font-weight: 600; color: #0F172A;">Recommended Model:</span>
                    <strong>Model 1 (ResNet101 Gaussian + ROS)</strong> is prioritized as the study benchmark. It demonstrated superior diagnostic reliability on unseen test cases with an overall accuracy of <strong>86.97%</strong>, a sensitivity of <strong>86.97%</strong>, and a specificity of <strong>92.10%</strong>.
                </div>

                <div style="margin-bottom: 0.75rem; line-height: 1.6; font-size: 0.92rem; color: #334155;">
                    <span style="font-weight: 600; color: #0F172A;">Clinical Interpretation:</span>
                    The model classifies lesion <em>"{patient_name}"</em> as 
                    <strong style="color: {badge_color};">{diag_m1}</strong> with <strong>{conf1:.1f}%</strong> certainty. The integrated 5×5 Gaussian spatial filtering removes superficial skin artifact noise and illumination variances, enhancing the network's focus on subtle border irregularities.
                </div>

                <div style="
                    border-top: 1px dashed #E2E8F0;
                    padding-top: 0.6rem;
                    font-size: 0.78rem;
                    color: #64748B;
                    line-height: 1.4;
                ">
                    <strong>⚠️ Clinical Governance Notice:</strong> This platform is designed solely to support professional medical workflow triage. Findings must be correlated with clinical examination, dermatoscopy pattern analysis, and histopathological biopsy.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 2: DATABASE AUDIT TRAIL ====================
with tab_records:
    st.subheader("Diagnostic Database Records & Patient History")
    st.caption("Inspect prior evaluations, patient names, and review uploaded lesion images.")
    
    df_history = db.get_user_history(st.session_state["username"])
    
    if not df_history.empty:
        st.dataframe(df_history, width="stretch", hide_index=True)
        
        st.markdown("---")
        st.subheader("Inspect Historical Record Image")
        
        selected_id = st.selectbox("Select Record ID to view uploaded image:", options=df_history["id"].tolist())
        
        if selected_id:
            img_blob = db.get_image_by_id(selected_id)
            if img_blob:
                import io
                image_stream = io.BytesIO(img_blob)
                stored_image = Image.open(image_stream)
                st.image(stored_image, caption=f"Record ID #{selected_id} Lesion Image", width=250)
            else:
                st.warning("No image data found for this record.")
        
        csv_file = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Historical Log (.CSV)",
            data=csv_file,
            file_name=f"diagnostic_history_{st.session_state['username']}.csv",
            mime="text/csv"
        )
    else:
        st.info("No inference evaluations have been recorded in the database yet.")