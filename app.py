import streamlit as st
import PIL
import cv2
import numpy as np
import utils
import io

st.set_page_config(
    page_title="Age/Gender/Emotion",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title('Age/Gender/Emotion :sun_with_face:')

st.sidebar.header('Type')
source_radio = st.sidebar.radio('Select Source', ['IMAGE', 'VIDEO', 'WEBCAM'])

if source_radio == 'IMAGE':
    st.sidebar.header('Upload')
    input = st.sidebar.file_uploader('Choose an image.', type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv)
        st.image(visualized_image, channels='BGR')
    else:
        st.image('assets/sample_image.jpg')
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")

elif source_radio == 'VIDEO':
    st.sidebar.header('Upload')
    input = st.sidebar.file_uploader('Choose a video.', type=("mp4"))

    if input is not None:
        temporary_location = 'upload.mp4'
        with open(temporary_location, 'wb') as out:
            out.write(input.read())  # Write file content once

        st.write(f"Attempting to play video from {temporary_location}")
        play_video(temporary_location)
    else:
        st.video('assets/sample_video.mp4')
        st.write("Click on 'Browse Files' in the sidebar to run inference on a video.")

elif source_radio == 'WEBCAM':
    st.write("Attempting to play video from webcam")
    play_video(0)  # Make sure play_video handles webcam input properly


