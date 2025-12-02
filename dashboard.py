import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time

import pandas as pd
from datetime import datetime

def log_violation(label):
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "violation": label
    }])
    # Append to CSV in project root
    df.to_csv("violations.csv", mode="a", header=not pd.io.common.file_exists("violations.csv"), index=False)

# Load your trained model
model = YOLO("weights/best.pt")


## show all the parameters of the model for debugging
# print("MODEL CLASS MAP:", model.names)

# Streamlit page config
st.set_page_config(page_title="PPEye Dashboard", layout="wide")

# ------------------------- CSS FOR PERFECT ALIGNMENT & THEME -------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 0rem;
        }
        .title-text {
            font-size: 42px;
            font-weight: 900;
            color: black;
            text-align: center;
            margin-bottom: -5px;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            color: black;
            margin-top: -10px;
        }
        .stButton>button {
            background-color: black !important;
            color: #FFD84D !important;
            font-size: 18px !important;
            padding: 10px 25px !important;
            border-radius: 10px !important;
            border: none !important;
        }
        .stButton>button:hover {
            background-color: #333 !important;
            color: white !important;
        }
        .info-card {
            background-color: #00000010;
            padding: 15px;
            border-radius: 12px;
            border-left: 6px solid black;
            margin-bottom: 10px;
        }
        .centered {
            display: flex;
            justify-content: center;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------- HEADER -------------------------
st.markdown('<p class="title-text">PPEye — Watching Every Worker, Every Moment</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time PPE Detection Dashboard</p>', unsafe_allow_html=True)

st.write("")

# ------------------------- ABOUT CARD -------------------------
st.markdown("""
<div class="info-card">
    <h4 style="margin-bottom:3px; color:black;">🔍 About This Dashboard</h4>
    <p style="margin-top:5px; color:black;">
        This interface provides real-time monitoring of workers using your custom-trained YOLOv11s PPE detection model. 
        The system identifies helmets, vests, and detects violations like <b>No Helmet</b> or <b>No Vest</b>. 
        Everything runs locally on your device with fast inference and a clean industrial-themed UI.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------- STATES -------------------------
if "detecting" not in st.session_state:
    st.session_state.detecting = False

# ------------------------- BUTTON ROW -------------------------
btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])

with btn_col2:
    start = st.button("Start Real-Time Detection")

with btn_col2:
    stop = st.button("Stop Detection")

if start:
    st.session_state.detecting = True
if stop:
    st.session_state.detecting = False

# ------------------------- SEE ANALYTICS BUTTON -------------------------
analytics_col1, analytics_col2, analytics_col3 = st.columns([2,1,2])

with analytics_col2:
    analytics_btn = st.button("📊 See Analytics")

if analytics_btn:
    st.switch_page("pages/analytics.py")


# ------------------------- VIDEO WINDOW -------------------------
frame_placeholder = st.empty()

# ------------------------- REAL-TIME LOOP -------------------------
if st.session_state.detecting:
    cap = cv2.VideoCapture(0)

    while st.session_state.detecting:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not detected!")
            break

        # Run YOLO on the frame
        results = model(frame)

        ###
        print("Detected classes:", results[0].boxes.cls.cpu().numpy())

        # Safe extraction of class ids (handles no detections)
        try:
            # results[0].boxes.cls is typically a torch tensor; convert safely
            if hasattr(results[0].boxes, "cls") and len(results[0].boxes) > 0:
                cls_tensor = results[0].boxes.cls
                try:
                    classes = cls_tensor.cpu().numpy().astype(int)
                except Exception:
                    classes = np.array(cls_tensor).astype(int)
            else:
                classes = np.array([], dtype=int)
        except Exception:
            classes = np.array([], dtype=int)

        # Log violations (avoid duplicate spam by logging per-frame; you can refine later)
        for c in classes:
            if c == 7:  # no_helmet 
                log_violation("No Helmet")
            elif c == 5:  # no_vest + helmet
                log_violation("No Vest")

        # Draw annotated frame
        annotated_frame = results[0].plot()

        # Resize to fit screen without scrolling (65%)
        annotated_frame = cv2.resize(annotated_frame, (0, 0), fx=0.65, fy=0.65)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display frame (no deprecated params, no warnings)
        frame_placeholder.image(annotated_frame)

        # small sleep so Streamlit can update UI smoothly
        time.sleep(0.03)

        if not st.session_state.detecting:
            break

    # release camera and clear placeholder after loop ends
    cap.release()
    frame_placeholder.empty()
