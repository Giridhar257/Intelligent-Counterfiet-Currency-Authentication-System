import sys
import os

# Fix import paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import pytesseract

from utils.text_features import extract_text_features
from models.multitask_model import MultiTaskModel
from utils.preprocessing import get_val_transforms
from utils.texture_features import extract_texture_features

st.set_page_config(
    page_title="Currency Detector",
    page_icon="💵",
    layout="centered"
)

st.title("💵 Multi-Country Currency Detection System")
st.write("Detect Country, Denomination & Fake/Real")

device = torch.device("cpu")

@st.cache_resource
def load_model():
    MODEL_PATH = os.path.join(BASE_DIR, "training", "cpu_model.pth")

    model = MultiTaskModel(num_denoms=9, num_countries=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

auth_map = {0: "REAL", 1: "FAKE"}

country_map = {
    0: "India 🇮🇳",
    1: "Indonesia 🇮🇩"
}

denom_map = {
    0: "10",
    1: "100",
    2: "20",
    3: "200",
    4: "2000",
    5: "50",
    6: "500",
    7: "100000",
    8: "50000"
}

def edge_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.mean()

uploaded_files = st.file_uploader(
    "Upload Currency Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        st.markdown("---")
        st.subheader(f"Processing: {uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        texture = extract_texture_features(image_bgr)
        texture = torch.tensor(texture, dtype=torch.float32).unsqueeze(0).to(device)

        transform = get_val_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)

        raw_text = pytesseract.image_to_string(image_np).upper()
        text_feat = extract_text_features(raw_text)
        text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            auth_out, denom_out, country_out = model(image_tensor, texture, text_tensor)

            auth_probs = torch.softmax(auth_out, dim=1)
            denom_probs = torch.softmax(denom_out, dim=1)
            country_probs = torch.softmax(country_out, dim=1)

        auth_pred = torch.argmax(auth_probs, dim=1).item()
        denom_pred = torch.argmax(denom_probs, dim=1).item()
        country_pred = torch.argmax(country_probs, dim=1).item()

        real_prob = auth_probs[0][0].item()
        fake_prob = auth_probs[0][1].item()

        suspicious_words = ["PROOF", "COPY", "SPECIMEN", "VOID", "TEST"]
        ocr_flag = any(word in raw_text for word in suspicious_words)

        edge_val = edge_score(image_bgr)
        edge_flag = edge_val < 20

        strong_fake = False

        if ocr_flag:
            strong_fake = True

        fake_score = fake_prob
        real_score = real_prob

        if edge_flag:
            fake_score += 0.2


        if strong_fake:
            is_fake = True
        elif fake_score > real_score:
            is_fake = True
        else:
            is_fake = False

        st.markdown(" Results")

        col1, col2 = st.columns(2)
        col1.metric("Real %", f"{real_prob*100:.2f}%")
        col2.metric("Fake %", f"{fake_prob*100:.2f}%")

        st.write(f"Country: **{country_map[country_pred]}**")

        denom_label = denom_map.get(denom_pred, "Unknown")

        if country_pred == 0:
            denom_display = f"₹{denom_label}"
        else:
            denom_display = f"Rp {denom_label}"

        st.write(f"Denomination: **{denom_display}**")


        if is_fake:
            st.error(" FINAL RESULT: FAKE")
        else:
            st.success(" FINAL RESULT: REAL")

        reasons = []

        if ocr_flag:
            reasons.append("Suspicious text detected")

        if edge_flag:
            reasons.append("Low structural detail")

        if fake_prob > real_prob:
            reasons.append("Model leaning FAKE")

        if reasons:
            st.warning("Reason(s): " + ", ".join(reasons))

        st.write(f"Confidence: {max(real_prob, fake_prob)*100:.2f}%")
        st.write(f"Edge Score: {edge_val:.2f}")