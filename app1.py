import streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.models import Model
import joblib
import tempfile

# Function to load the VGG16 model
def load_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    return model

# Feature extraction function for VGG16
def extract_vgg16_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = vgg16_preprocess_input(x)
    features = model.predict(x)
    features_flatten = features.reshape((features.shape[0], -1))
    return features_flatten

# Load the SVM model
def load_svm_model(model_path):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Paths to the models
svm_model_compressed_path = 'svm_model1.xz'
label_encoder_compressed_path = 'label_encoder.xz'

# Load the models
vgg16_model = load_vgg16_model()
svm_classifier = load_svm_model(svm_model_compressed_path)
label_encoder = load_svm_model(label_encoder_compressed_path)  # Reusing the same function for simplicity

st.title('Citrus Disease Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict button outside the uploaded_file condition
if st.button('Predict'):
    if uploaded_file is not None:
        st.write("Classifying...")

        # Save the uploaded image
        img_path = 'uploaded_image.jpg'
        with open(img_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Extract features using VGG16
        vgg16_features = extract_vgg16_features(img_path, vgg16_model)

        # Predict the class
        prediction = svm_classifier.predict(vgg16_features)

        # Use inverse_transform to get the class label
        predicted_class = prediction[0]
        confidence = sigmoid(np.max(svm_classifier.decision_function(vgg16_features))) * 100

        # Display predicted class with bold and increased font size
        st.markdown(f'**Predicted Class:** {predicted_class}', unsafe_allow_html=True)
        
        # Display confidence value formatted to two decimal points
        st.write(f'Confidence: {confidence:.2f} %')

# Display note about uploading a citrus leaf image
st.write("Note: Make sure to upload an image of a citrus leaf.")
