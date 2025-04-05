import streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import joblib
import tempfile

# Cache VGG16 model with reduced output dimension
@st.cache_resource
def load_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Cache model loader for SVM and LabelEncoder
@st.cache_resource
def load_svm_model(model_path):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

# Feature extraction using VGG16
def extract_vgg16_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = vgg16_preprocess_input(x)
    features = model.predict(x)
    return features

# Sigmoid for confidence estimation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load models once (cached)
vgg16_model = load_vgg16_model()
svm_classifier = load_svm_model('svm_model1.xz')
label_encoder = load_svm_model('label_encoder.xz')

# UI
st.title('🍊 Citrus Disease Detection')

uploaded_file = st.file_uploader("Upload a citrus leaf image...", type=["jpg", "jpeg", "png"])

if st.button('Predict'):
    if uploaded_file is not None:
        st.write("🔍 Classifying...")

        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_img_path = tmp_file.name

        # Extract features and predict
        vgg16_features = extract_vgg16_features(tmp_img_path, vgg16_model)
        prediction = svm_classifier.predict(vgg16_features)

        # Confidence score
        confidence = sigmoid(np.max(svm_classifier.decision_function(vgg16_features))) * 100

        predicted_class = prediction[0]

        # Output results
        st.markdown(f'<h3>✅ Predicted Class: <strong>{predicted_class}</strong></h3>', unsafe_allow_html=True)
        st.markdown(f"<h4>🧠 Confidence: <strong>{confidence:.2f} %</strong></h4>", unsafe_allow_html=True)
    else:
        st.warning("Please upload an image before clicking Predict.")

# Helpful note
st.info("📌 Note: Upload a clear image of a citrus fruit for accurate results.")
