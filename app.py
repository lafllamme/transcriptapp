from io import BytesIO
from re import M
from blinker import base
import speech_recognition as sr
import streamlit as st
import pandas as pd
import numpy as np

import docx2txt
import pdfplumber
import time
import os
import base64
import io
import codecs

from typing import List
from pydub import AudioSegment
from pathlib import Path
from PIL import Image

from PyPDF2 import PdfFileReader
from PIL import Image
from os import path
from pathlib import Path
from base64 import b64decode

from streamlit.elements import form

# example sound
filename = "sound.wav"

# initialize the speech recognizer
r = sr.Recognizer()


def save_uploadedfile(uploadedfile, **kwargs):
    dataType = kwargs.get('dataType')
    print('Datatype:', dataType, 'FILE: ',  uploadedfile)
    if(dataType == 'image'):
        with open(os.path.join("tempDir/images", uploadedfile.name), "wb") as f:
            print('This is IMAGES')
            f.write(uploadedfile.getbuffer())

    elif(dataType == 'csv'):
        with open(os.path.join("tempDir/datasets", uploadedfile.name), "wb") as f:
            print('This is DATASETS')
            f.write(uploadedfile.getbuffer())

    elif(dataType == 'document'):
        with open(os.path.join("tempDir/documents", uploadedfile.name), "wb") as f:
            print('This is DOCS')
            f.write(uploadedfile.getbuffer())

    else:
        with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
            print('This is TEMPDIR')
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


def load_image(image_file):
    img = Image.open(image_file)
    return img


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


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

                    save_uploadedfile(image_file, dataType="image")
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
                        save_uploadedfile(data_file, dataType="csv")
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
                            save_uploadedfile(docx_file, dataType="document")
                            st.write(raw_text)
                            st.success('Done!')

                        # pdf
                        elif docx_file.type == "application/pdf":
                            try:
                                with pdfplumber.open(docx_file) as pdf:
                                    page = pdf.pages[0]
                                    time.sleep(2)
                                    save_uploadedfile(
                                        docx_file, dataType="document")
                                    st.write(page.extract_text())
                                    st.success('Done!')

                            except:
                                st.write("None")
                        # docx
                        elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or docx_file.type == 'application/msword':
                            raw_text = docx2txt.process(docx_file)
                            time.sleep(2)
                            save_uploadedfile(docx_file, dataType="document")
                            st.write(raw_text)
                            st.success('Done!')

    # my files menu
    if (choice == menu[1]):
        col1, col2, col3 = st.columns([4, 1, 1])
        fileList = []
        filePaths = []
        extensionList = []
        fileId = 0
        index = 0
        flag = False

        for root, dirs, files in os.walk("tempDir"):
            for file in files:
                fileName = os.path.join(file)
                extension = os.path.splitext(file)[1]
                fileList.append(fileName)
                extensionList.append(extension)

                if extension == '.csv':
                    stringPath = "tempDir/datasets/{}".format(fileName)
                    filePaths.append(stringPath)

                elif extension == '.pdf' or extension == '.docx' or extension == '.txt':
                    stringPath = "tempDir/documents/{}".format(fileName)
                    filePaths.append(stringPath)

                elif extension == '.jpeg' or extension == '.jpg' or extension == '.png':
                    stringPath = "tempDir/images/{}".format(fileName)
                    filePaths.append(stringPath)

                elif extension == '.mp3' or extension == '.wav':
                    stringPath = "tempDir/audios/{}".format(fileName)
                    filePaths.append(stringPath)
        fileList.sort()
        selectedItem = st.selectbox("Search 🗂️", fileList)
        matching = [s for s in filePaths if selectedItem in s]
        string = ' '.join(matching)

        with st.expander(selectedItem):
            fileDir = os.path.dirname(os.path.realpath('__file__'))
            filename = os.path.join(fileDir, string)
            print(fileDir, filename)
            agree = st.checkbox('Display Data 📊', key=selectedItem)
            if agree:
                if filename.endswith('csv'):
                    df = pd.read_csv(filename)
                    st.write(df)
                    csv = convert_df(df)
                    st.download_button(
                        label="Download Dataset",
                        data=csv,
                        file_name=selectedItem
                    )
                elif filename.endswith('jpg') or filename.endswith('png') or filename.endswith('jpeg'):
                    img = Image.open(filename)
                    st.image(img)
                    with open(filename, "rb") as img_file:
                        btn = st.download_button(
                            label="Download Image",
                            data=img_file,
                            file_name=selectedItem,
                        )

                elif filename.endswith('wav') or filename.endswith('mp3'):
                    st.audio(filename)
                    with open(filename, "rb") as audio_file:
                        btn = st.download_button(
                            label="Download Audio",
                            data=audio_file,
                            file_name=selectedItem

                        )

                elif filename.endswith('docx') or filename.endswith('pdf') or filename.endswith('txt'):
                    with open(filename, "rb") as text_file:
                        if filename.endswith('txt'):
                            st.text(str(text_file.read(), "utf-8"))

                        elif filename.endswith('pdf'):
                            encodedPdf = base64.b64encode(
                                text_file.read()).decode('utf-8')

                            pdfReader = PdfFileReader(text_file)
                            count = pdfReader.numPages
                            all_page_text = ""
                            for i in range(count):
                                page = pdfReader.getPage(i)
                                all_page_text += page.extractText()

                            flag = True

                            bytes = b64decode(encodedPdf, validate=True)

                            styl = f"""
                                    <style>
                                    .css-ns78wr {{
                                    display: inline-flex;
                                    -webkit-box-align: center;
                                    align-items: center;
                                    -webkit-box-pack: center;
                                    justify-content: center;
                                    font-weight: 400;
                                    padding: 0.25rem 0.75rem;
                                    border-radius: 0.25rem;
                                    margin: 0px;
                                    line-height: 1.6;
                                    color: inherit;
                                    width: auto;
                                    user-select: none;
                                    background-color: rgb(255, 255, 255);
                                    border: 1px solid rgba(49, 51, 63, 0.2);
                                    text-decoration: none;}}
                                    a {{ color: rgb(255, 75, 75); text-decoration: none;}}
                                    a:hover {{ text-decoration: none;}}
                                    .css-177yq5e a {{color: #383838;}}
                                    </style>
                                    <a href="data:application/octet-stream;base64,{encodedPdf}" class="css-ns78wr" download="{selectedItem}">Download PDF</a>
                                    """

                            st.write(all_page_text)
                            st.markdown(styl, unsafe_allow_html=True)

                            print(text_file)

                        elif filename.endswith('docx'):
                            raw_text = docx2txt.process(text_file)
                            st.write(raw_text)

                        if flag == False:
                            btn = st.download_button(
                                label="Download Document",
                                data=text_file,
                                file_name=selectedItem

                            )

        with st.expander("Raw Output"):
            st.write(filePaths)
            st.write(fileList)
            st.write(extensionList)

    if (choice == menu[3]):
        st.subheader("About")
        st.info("Dogan Teke, 7335741")
        st.info("Universität zu Köln, 2022")


if __name__ == '__main__':
    main()
