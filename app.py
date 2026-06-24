import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model("Brain_Tumor_CNN_Model.h5")

# ==========================
# CLASS LABELS
# ==========================
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("🧠 Project Information")

st.sidebar.header("Project Description")

st.sidebar.write(
    """
    This system uses a Convolutional Neural Network (CNN)
    to classify Brain MRI images into four categories:
    
    • Glioma
    
    • Meningioma
    
    • No Tumor
    
    • Pituitary
    """
)

st.sidebar.header("Model Performance")

st.sidebar.metric(
    label="Validation Accuracy",
    value="84.35%"
)

st.sidebar.info(
    """
    Model Used: CNN
    
    Image Size: 224 × 224
    
    Framework: TensorFlow/Keras
    
    Frontend: Streamlit
    """
)

# ==========================
# MAIN TITLE
# ==========================
st.markdown(
    """
    # 🧠 Brain Tumor Detection System

    ### AI-Powered MRI Analysis using Deep Learning

    Upload an MRI scan image to identify the tumor type.
    """
)

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# PREDICTION SECTION
# ==========================
if uploaded_file is not None:

    st.subheader("📷 Uploaded MRI Scan")

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    try:

        # ==========================
        # PREPROCESS IMAGE
        # ==========================
        img = image.convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ==========================
        # MODEL PREDICTION
        # ==========================
        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        result = class_names[predicted_class]

        confidence = np.max(prediction) * 100

        # ==========================
        # RESULT SECTION
        # ==========================
        st.subheader("🔍 Prediction Result")

        st.success(f"Prediction: {result}")

        # ==========================
        # TUMOR INFORMATION
        # ==========================
        if result == "Glioma":

            st.info(
                """
                Glioma is a tumor that develops in the brain
                and spinal cord from glial cells.
                """
            )

        elif result == "Meningioma":

            st.info(
                """
                Meningioma develops in the membranes
                surrounding the brain and spinal cord.
                """
            )

        elif result == "Pituitary":

            st.info(
                """
                Pituitary tumors develop in the pituitary gland,
                which controls hormone production.
                """
            )

        elif result == "No Tumor":

            st.success(
                """
                No tumor detected in the MRI image.
                """
            )

        # ==========================
        # CONFIDENCE SCORE
        # ==========================
        st.subheader("📊 Confidence Score")

        st.write(f"Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

        # ==========================
        # CLASS PROBABILITIES
        # ==========================
        st.subheader("📈 Prediction Probabilities")

        for i, class_name in enumerate(class_names):

            prob = float(prediction[0][i])

            st.write(
                f"{class_name}: {prob * 100:.2f}%"
            )

            st.progress(
                int(prob * 100)
            )

    except Exception as e:

        st.error(
            f"Error during prediction: {e}"
        )

# ==========================
# FOOTER
# ==========================
st.markdown("---")

st.caption(
    "Developed by Madhumithra | Brain Tumor Detection using CNN and Streamlit"
)