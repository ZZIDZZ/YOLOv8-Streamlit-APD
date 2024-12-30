#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Description:
-------------------------------------------------
"""
from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Deteksi Alat Pelindung Diri Konstruksi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Deteksi Alat Pelindung Diri Konstruksi")

# sidebar
st.sidebar.header("Konfigurasi Model")

# model options
task_type = st.sidebar.selectbox(
    "Task",
    ["Deteksi"]
)

model_type = None
if task_type == "Deteksi":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Saat ini hanya fungsi 'Deteksi' yang diimplementasikan")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Pilih model terlebih dahulu")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Tidak dapat memuat model: {e}")

# image/video options
st.sidebar.header("Konfigurasi Citra")
source_selectbox = st.sidebar.selectbox(
    "Pilih Sumber",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Saat ini hanya video, gambar, dan webcam yang diimplementasikan")