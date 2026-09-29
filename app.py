import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms

st.set_page_config(
    page_title="Bharatanatyam Mudra Recognition",
    page_icon="🤲",
    layout="centered",
)

MODEL_PATH = Path(__file__).resolve().parent / "model" / "mudra_cnn.pth"
CLASS_PATH = Path(__file__).resolve().parent / "model" / "class_names.json"


class MudraCNN(nn.Module):
    """Small custom CNN used for the five mudra classes."""

    def __init__(self, num_classes, dropout=0.30):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


@st.cache_resource
def load_model():
    # The checkpoint is stored with the exact class order used in training.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = MudraCNN(len(class_names), dropout=0.30)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def prepare_image(image):
    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    tensor = transform(image).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def predict_image(image):
    model, class_names = load_model()
    tensor = prepare_image(image)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].numpy()

    predicted_index = int(np.argmax(probabilities))
    return (
        class_names[predicted_index],
        float(probabilities[predicted_index]),
        probabilities,
        class_names,
    )


def show_home():
    st.title("Bharatanatyam Mudra Recognition")
    st.write(
        "A simple CNN-based classifier that recognizes five Bharatanatyam hand mudras from an image."
    )

    st.subheader("What this project does")
    st.write(
        "Upload a practice image and the model predicts one of five mudras, "
        "along with its confidence and the probabilities for all classes."
    )

    st.subheader("Classes")
    st.write("Musti, Pataka, Sikhara, Simhamukha and Trisula")

    st.info(
        "The supplied dataset is visually consistent, so the clean test score is very high. "
        "For real practice use, the model should be tested on new student images from different cameras and backgrounds."
    )


def show_predict():
    st.title("Predict a Mudra")
    st.write("Upload a JPG or PNG image of a hand mudra.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be read as an image.")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    predicted_mudra, confidence, probabilities, class_names = predict_image(image)

    st.subheader(f"Predicted mudra: {predicted_mudra}")
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    if confidence < 0.70:
        st.warning(
            "The model is not very confident. Try a clearer image with the complete hand visible."
        )
    else:
        st.success("The model has a reasonably strong prediction for this image.")

    st.subheader("Probability distribution")
    probability_df = pd.DataFrame(
        {
            "Mudra": class_names,
            "Probability": probabilities,
        }
    ).set_index("Mudra")

    st.bar_chart(probability_df)
    st.dataframe(
        probability_df.assign(
            Probability=probability_df["Probability"].map(lambda value: f"{value * 100:.2f}%")
        ),
        use_container_width=True,
    )


def show_about():
    st.title("About the Project")

    st.subheader("Dataset")
    st.write("2,000 RGB images, 400 per class, with five mudra classes.")
    st.write("Original image size: 300 × 300 pixels")
    st.write("Train / validation / test split: 70% / 15% / 15%")

    st.subheader("Model")
    st.write(
        "Custom CNN with three convolution blocks, batch normalization, ReLU activation, "
        "max pooling and a small fully connected classifier."
    )
    st.write("Input size: 64 × 64")
    st.write("Selected learning rate: 0.001")
    st.write("Selected dropout: 0.30")

    st.subheader("Evaluation")
    st.write("Clean test accuracy on the supplied split: 100%")
    st.write(
        "A separate occlusion stress test reached about 67% accuracy, showing that performance can drop "
        "when important finger information is hidden."
    )

    st.subheader("Useful next feature")
    st.write(
        "A practice mode with a reference mudra image and webcam-based feedback would make the app much more useful for dance learners."
    )


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

if page == "Home":
    show_home()
elif page == "Predict":
    show_predict()
else:
    show_about()
