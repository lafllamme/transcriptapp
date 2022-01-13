import speech_recognition as sr
import streamlit as st
import pandas as pd
import numpy as np

import docx2txt
import pdfplumber
import time
import os

from typing import List
from pydub import AudioSegment
from pathlib import Path
from PIL import Image
from PyPDF2 import PdfFileReader
from PIL import Image



# example sound
filename = "sound.wav"

# initialize the speech recognizer
r = sr.Recognizer()


def save_uploadedfile(uploadedfile):
    dataType = uploadedfile.type
    if(dataType == 'image/png' or 'image/jpeg' or 'image/jpeg'):
        with open(os.path.join("tempDir/images", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    if(dataType == 'text/csv'):
        with open(os.path.join("tempDir/datasets", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    if(dataType == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or 'text/plain' or 'application/pdf'):
        with open(os.path.join("tempDir/documents", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    else:
        with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    return st.success("Saved File: {} to tempDir".format(uploadedfile.name))


def upload_and_save_wavfiles(save_dir: str, **kwargs) -> List[Path]:
    uploaded_files = kwargs.get('uploader')
    save_paths = []

    try:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:

                if uploaded_file.name.endswith('wav'):
                    audio = AudioSegment.from_wav(uploaded_file)
                    file_type = 'wav'

                elif uploaded_file.name.endswith('mp3'):
                    audio = AudioSegment.from_mp3(uploaded_file)
                    file_type = 'mp3'

            save_path = Path(save_dir) / uploaded_file.name
            save_paths.append(save_path)
            audio.export(save_path, format=file_type)
        return save_paths

    except TypeError:
        col1, col2, col3 = st.columns([3, 6, 2])
        with col1:
            st.write("")

        with col2:
            st.markdown(
                "![Done Loading](https://media4.giphy.com/media/26u4lOMA8JKSnL9Uk/giphy.gif)")

        with col3:
            st.write("")


def display_wavfile(wavpath):
    audio_bytes = open(wavpath, 'rb').read()
    file_type = Path(wavpath).suffix
    st.audio(audio_bytes, format=f'audio/{file_type}', start_time=0)
    files = upload_and_save_wavfiles('tempDir/audios')


def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text


def read_pdf_with_pdfplumber(file):
    with pdfplumber.open(file) as pdf:
        page = pdf.pages[0]
        return page.extract_text()


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():

    # app cover
    st.set_page_config(
        page_title="✍🏻 Transcriptor",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded")

    col1, col2, col3 = st.columns([3, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.title("🤖 Transcriptor App")
    with col3:
        st.write("")

    # sidebar menu
    menu = ["📂 Upload", "🔎 My Files", "💬 Transcript", "ℹ️ About"]
    choice = st.sidebar.selectbox("⚙️ Operations", menu)

    # upload menu
    if (choice == menu[0]):
        # audiofiles
        st.subheader("Audio")
        audioFile = st.file_uploader(
            "Choose Audio Format", type=['wav', 'mp3'], accept_multiple_files=True, key="audio_file_uploader")
        files = upload_and_save_wavfiles(
            save_dir='tempDir/audios', uploader=audioFile)
        for wavpath in files:
            with st.spinner('Uploading Files...'):
                time.sleep(2)
                display_wavfile(wavpath)
                st.success('Done!')

        with st.expander("Upload Images"):
            # images
            st.subheader("Images")
            image_file = st.file_uploader(
                "Choose Image", type=['png', 'jpeg', 'jpg'], key="image_file_uploader")
            if image_file is not None:
                with st.spinner('Uploading Files...'):
                    file_details = {"Filename": image_file.name,
                                    "FileType": image_file.type, "FileSize": image_file.size}
                    st.write(file_details)
                    time.sleep(2)
                    save_uploadedfile(image_file)
                    img = load_image(image_file)
                    st.image(img)
                    st.success('Done!')

        with st.expander("Upload Datasets (CSV)"):
            # datasets
            st.subheader("Dataset")
            data_file = st.file_uploader(
                "Choose CSV", type=['csv'], key="dataset_file_uploader")
            if st.button("Upload File", key="dataset_button"):
                if data_file is not None:
                    with st.spinner('Uploading Files...'):

                        file_details = {"Filename": data_file.name,
                                        "FileType": data_file.type, "FileSize": data_file.size}
                        st.write(file_details)
                        time.sleep(2)
                        save_uploadedfile(data_file)
                        df = pd.read_csv(data_file)
                        st.dataframe(df)
                        st.success('Done!')

        with st.expander("Upload Documents"):
            # documents
            st.subheader("Documents")
            docx_file = st.file_uploader(
                "Choose Doc", type=['txt', 'docx', 'pdf'], key="document_file_uploader")
            if st.button("Upload File", key="document_button"):
                if docx_file is not None:
                    file_details = {"Filename": docx_file.name,
                                    "FileType": docx_file.type, "FileSize": docx_file.size}
                    st.write(file_details)

                    with st.spinner('Uploading Files...'):
                        # plain text
                        if docx_file.type == "text/plain":
                            st.text(str(docx_file.read(), "utf-8"))
                            raw_text = str(docx_file.read(), "utf-8")
                            time.sleep(2)
                            save_uploadedfile(docx_file)
                            st.write(raw_text)
                            st.success('Done!')

                        # pdf
                        elif docx_file.type == "application/pdf":
                            try:
                                with pdfplumber.open(docx_file) as pdf:
                                    page = pdf.pages[0]
                                    time.sleep(2)
                                    save_uploadedfile(docx_file)
                                    st.write(page.extract_text())
                                    st.success('Done!')

                            except:
                                st.write("None")
                        # docx
                        elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            raw_text = docx2txt.process(docx_file)
                            time.sleep(2)
                            save_uploadedfile(docx_file)
                            st.write(raw_text)
                            st.success('Done!')

    # my files menu
    if (choice == menu[1]):
        col1, col2, col3 = st.columns([4, 1, 1])
        fileList = []
        fileId = 0

        for root, dirs, files in os.walk("tempDir"):
            for file in files:
                filePath = os.path.join(root, file)
                fileName = os.path.join(file)
                fileList.append(fileName)
        for item in fileList:
            fileKey = "fileKey_{}".format(fileId)
            with st.expander(item):
                f = open(filePath   , "wb")
                agree = st.checkbox('Display Data 📊', key=fileKey)
                if agree:
                    st.write(f)
                    st.download_button("Download", item, key=fileKey)
                fileId += 1

        st.selectbox("Select Item", fileList)
        with st.expander("Raw Output"):
            st.write(fileList)

    if (choice == menu[3]):
        st.subheader("About")
        st.info("Dogan Teke, 7335741")
        st.info("Universität zu Köln, 2022")


if __name__ == '__main__':
    main()
