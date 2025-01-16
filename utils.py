#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

class YOLOv8EigenCAM:
    def __init__(self, model, target_layers):
        """
        Initialize EigenCAM with YOLOv8 model and target layers.
        :param model: Pre-trained YOLOv8 model.
        :param target_layers: List of target layers for EigenCAM.
        """
        self.model = model
        self.target_layers = target_layers
        self.eigen_cam = EigenCAM(model=model, target_layers=target_layers, task='od')

    def generate_cam(self, img_tensor):
        """
        Generate EigenCAM heatmap and process for visualization.
        :param img_tensor: Input tensor for the model.
        :return: Processed Grad-CAM heatmap.
        """
        # Generate EigenCAM
        grayscale_cam = self.eigen_cam(img_tensor)[0, :, :]  # Extract grayscale CAM
        return grayscale_cam


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

@st.cache_resource
def load_model_with_eigencam(model_path, target_layer_idx=-4):
    """
    Load YOLOv8 model and set up EigenCAM.
    :param model_path: Path to the YOLOv8 model file.
    :param target_layer_idx: Index of the target layer for EigenCAM.
    :return: YOLO model and EigenCAM instance.
    """
    model = YOLO(model_path)
    target_layers = [model.model.model[target_layer_idx]]
    grad_cam = YOLOv8EigenCAM(model, target_layers)
    return model, grad_cam



def infer_uploaded_image(conf, model):
    """
    Execute inference for Citra Awal
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Pilih Gambar...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the Citra Awal to the page with caption
            st.image(
                image=source_img,
                caption="Citra Awal",
                use_container_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image, conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Citra Terdeteksi",
                             use_container_width=True)
                    
                    try:
                        # Display detected objects and their counts
                        with st.container():
                            st.write("Objek Terdeteksi:")
                            detected_classes = [model.names[int(cls)] for cls in boxes.cls]
                            class_counts = {cls: detected_classes.count(cls) for cls in set(detected_classes)}
                            
                            for cls, count in class_counts.items():
                                if("no-" not in cls):
                                    st.write(f"{cls}: {count}")

                            st.write(":red[Pelanggaran Terdeteksi:]")
                            for cls, count in class_counts.items():
                                if("no-" in cls):
                                    st.write(f":red[{cls}: {count}]")
                    
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)



def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

def letterbox(image, new_shape=(640, 640), color=(0, 0, 0)):
    """
    Resize an image to fit the desired shape with letterboxing, keeping aspect ratio.
    :param image: Input image (numpy array).
    :param new_shape: Target shape (width, height).
    :param color: Padding color.
    :return: Resized and padded image.
    """
    shape = image.shape[:2]  # Current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate the scaling factor
    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    scaled_shape = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))

    # Resize the image
    resized_image = cv2.resize(image, scaled_shape, interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    dw = new_shape[1] - scaled_shape[0]  # Width padding
    dh = new_shape[0] - scaled_shape[1]  # Height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    # Add padding
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_image


def infer_uploaded_image_with_gradcam(conf, model, grad_cam):
    """
    Execute inference for Citra Awal with EigenCAM visualization.
    :param conf: Confidence threshold for YOLOv8 model.
    :param model: YOLOv8 model instance.
    :param grad_cam: EigenCAM instance.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Pilih Gambar...",
        type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            st.image(
                image=source_img,
                caption="Citra Awal",
                use_container_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                # Convert image to OpenCV format and resize
                # Convert image to OpenCV format and resize with letterbox
                image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
                # image_resized = letterbox(image, new_shape=(640, 640))
                image_resized = cv2.resize(image, (640, 640))
                # img_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_tensor = image_resized.copy()


                # Model prediction
                res = model.predict(img_tensor, conf=conf)
                res_plotted = res[0].plot()[:, :, ::-1]

                # Generate EigenCAM heatmap
                try:
                    grayscale_cam = grad_cam.generate_cam(img_tensor)  # Generate grayscale CAM
                    cam_image = show_cam_on_image(image_resized / 255.0, grayscale_cam, use_rgb=True)
                except Exception as e:
                    cam_image = None
                    st.warning(f"EigenCAM could not be generated: {e}")

                with col2:
                    # concat 2 images left and right using Image
                    st.image(Image.fromarray(np.hstack((res_plotted, cam_image))),
                             caption="Citra Terdeteksi & Overlay EigenCAM",
                             use_container_width=True)
                    # st.image(res_plotted, caption="Citra Terdeteksi", use_container_width=True)
                    # if cam_image is not None:
                        # st.image(cam_image, caption="EigenCAM Overlay", use_container_width=True)
