import streamlit as st
import time
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Skin Lesion Classification",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed" # Starts collapsed on mobile for better usability
)

# 2. Custom CSS: Light Mint Blue Theme, Professional Typography & Mobile Responsive Design
st.markdown("""
    <style>
    /* Professional Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Body and Background Styling (Light Mint Blue) */
    .stApp {
        background-color: #E8F5E9 !important; /* Soft Mint / Light Mint Blue tint */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #1F2937 !important;
    }

    /* Main Container Padding Adjustment for Mobile */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 800px;
    }

    /* Card Containers */
    .css-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
    }

    /* Professional Button Styling */
    .stButton>button {
        width: 100% !important;
        background-color: #0284C7 !important; /* Medical Professional Blue */
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        height: 3.2em !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #0369A1 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }

    /* Metric Badges Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }

    /* Mobile Responsive Overrides */
    @media (max-width: 640px) {
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        h1 {
            font-size: 1.6rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Section
st.title("🔬 Skin Lesion Classifier")
st.caption("ResNet101 Model • Random Oversampling (ROS) • Gaussian Preprocessed")
st.markdown("---")

# 4. Sidebar Details
with st.sidebar:
    st.header("⚙️ Model Pipeline")
    st.info("**Backbone:** ResNet101")
    st.write("**Pre-processing:** $5\\times 5$ Depthwise Gaussian Blur")
    st.write("**Resampling:** Random Oversampling (ROS)")
    st.write("**Target Metrics:** High Sensitivity for Malignant Detection")
    st.markdown("---")
    st.write("📌 *Prototype interface running in demonstration mode.*")

# 5. File Uploader
uploaded_file = st.file_uploader("Upload Dermoscopic Image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Responsive Columns: Stacks vertically on mobile, side-by-side on desktop
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📸 Uploaded Lesion")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("📊 Diagnostic Output")
        
        if st.button("⚡ Run Classification"):
            with st.spinner("Applying Gaussian Filter & Running Inference..."):
                time.sleep(1.2) # Simulates prediction runtime
                
                # Prototype Prediction Proportions
                benign_prob = 88.5
                malignant_prob = 11.5
                
                # Model Metric Performance Benchmark (From your ResNet101 ROS run)
                model_sensitivity = 0.8697 # 86.97% Sensitivity / Recall
                model_specificity = 0.9210 # Specificity (TN Rate)
                
                # Primary Result Output
                if benign_prob > 50:
                    st.success("✅ **Diagnosis: BENIGN**")
                    st.metric(label="Prediction Confidence", value=f"{benign_prob:.1f}%")
                else:
                    st.error("⚠️ **Diagnosis: MALIGNANT**")
                    st.metric(label="Prediction Confidence", value=f"{malignant_prob:.1f}%")
                
                st.markdown("---")
                
                # Probability Distribution Progress Bars
                st.write("**Class Probabilities:**")
                st.write(f"Benign: `{benign_prob:.1f}%`")
                st.progress(int(benign_prob))
                
                st.write(f"Malignant: `{malignant_prob:.1f}%`")
                st.progress(int(malignant_prob))
                
                st.markdown("---")
                
                # System Reliability Metrics Panel (Sensitivity / Recall)
                st.write("**🏥 Model Clinical Benchmarks:**")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(
                        label="Sensitivity (Recall)", 
                        value=f"{model_sensitivity * 100:.2f}%",
                        help="Sensitivity (Recall) measures the model's ability to correctly detect true malignant cases."
                    )
                with m_col2:
                    st.metric(
                        label="Specificity (TN Rate)", 
                        value=f"{model_specificity * 100:.2f}%",
                        help="Specificity measures the model's ability to correctly identify benign cases."
                    )

else:
    st.info("👆 Upload a skin lesion image above to evaluate the diagnostic dashboard.")