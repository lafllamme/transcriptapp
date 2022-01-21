
from asyncore import write
import speech_recognition as sr
import streamlit as st
import pandas as pd
import numpy as np
import docx2txt
import pdfplumber
import time
import os
import base64
import requests
import torch

from re import search
from pathlib import Path
from typing import List
from pydub import AudioSegment
from pathlib import Path
from PIL import Image
from PyPDF2 import PdfFileReader
from PIL import Image
from os import path
from pathlib import Path
from base64 import b64decode
from streamlit import StreamlitAPIException
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData
from streamlit.elements import form
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import AutoModel, TFAutoModel, AutoTokenizer
from genericpath import exists
from io import BytesIO
from re import M, X
from blinker import base

from generator import *
from helpers import *

def main():

    # app cover
    st.set_page_config(
        page_title="✍🏻 Transcriptor",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded")

    foundFiles = []

    # ugly workaround because streamlit cloud doesn't support git lfs -.-
    for model in Path().cwd().glob("./*"):
        foundFiles.append(str(model))
    for files in Path().cwd().glob("distilbert-dlf/*"):
        foundFiles.append(str(files))

    checkFiles = ("distilbert-dlf/pytorch_model.bin",
                  "wandering-sponge-4.pth", "label_embeddings.npy")
    for path in checkFiles:
        if os.path.exists(path) == False:
            print('I miss :', path)
            msg = st.warning("🚩 Models need to be downloaded... ")
            try:
                with st.spinner('Initiating...'):
                    time.sleep(3)
                    url_pth = "https://www.dl.dropboxusercontent.com/s/wxbsve2tz8qqoha/wandering-sponge-4.pth?dl=0"
                    url_npy = "https://www.dl.dropboxusercontent.com/s/fse0o153tm4bwpp/label_embeddings.npy?dl=0"
                    url_bin = "https://www.dl.dropboxusercontent.com/s/qp20l5ryhcoavsx/pytorch_model.bin?dl=0"

                    r_pth = requests.get(url_pth, allow_redirects=True)
                    r_npy = requests.get(url_npy, allow_redirects=True)
                    r_bin = requests.get(url_bin, allow_redirects=True)

                    open("wandering-sponge-4.pth", 'wb').write(r_pth.content)
                    open("label_embeddings.npy", 'wb').write(r_npy.content)
                    open("distilbert-dlf/pytorch_model.bin",
                         'wb').write(r_bin.content)
                    del r_pth, r_npy, r_bin
                    msg.success("Download was successful ✅")
            except:
                msg.error("Error downloading model files...😥")

    colT1, colT2 = st.columns([3, 8])
    with colT2:
        st.title("🤖 Transcriptor App")

    # sidebar menu
    menu = ["📂 Upload", "🔎 My Files", "💬 Transcript", "ℹ️ About"]
    choice = st.sidebar.selectbox("⚙️ Operations", menu)

    # upload menu
    if (choice == menu[0]):
        # audiofiles
        c1, c2, c3 = st.columns([3, 4, 3])
        with c1:
            st.write("")
        with c2:
            st.error("⬆️ Upload the files you want to transcribe")
        with c3:
            st.write("")

        st.header("🎶 Audio")
        audioFile = st.file_uploader(
            "🔎 Choose Audio Format", type=['wav', 'mp3'], accept_multiple_files=True, key="audio_file_uploader")
        files = upload_and_save_wavfiles(
            save_dir='tempDir/audios', uploader=audioFile)
        for wavpath in files:
            with st.spinner('Uploading Files...'):
                time.sleep(2)
                display_wavfile(wavpath)
                st.success('Done!')

        with st.expander("🖼️ Upload Images"):
            # images
            st.subheader("Images")
            image_file = st.file_uploader(
                "🔎 Choose Image", type=['png', 'jpeg', 'jpg'], key="image_file_uploader")
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

        with st.expander("💾 Upload Datasets (CSV)"):
            # datasets
            st.subheader("Dataset")
            data_file = st.file_uploader(
                "🔎 Choose CSV", type=['csv'], key="dataset_file_uploader")
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

        with st.expander(" 📄 Upload Documents"):
            # documents
            st.subheader("Documents")
            docx_file = st.file_uploader(
                "🔎 Choose Doc", type=['txt', 'docx', 'pdf'], key="document_file_uploader")
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
        st.header("Search 🗂️")
        selectedItem = st.selectbox("", fileList)
        matching = [s for s in filePaths if selectedItem in s]
        string = ' '.join(matching)

        try:
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

                        btn_c1, btn_c2 = st.columns([1, 5])

                        with btn_c1:
                            st.download_button(
                                label="Download Dataset",
                                data=csv,
                                file_name=selectedItem
                            )
                        with btn_c2:
                            delete = st.button("Delete File")
                            if delete:
                                with st.spinner('Deleting...'):
                                    os.remove(filename)
                                    time.sleep(2)
                                    st.success('Success ✅')
                                    rerun()
                    elif filename.endswith('jpg') or filename.endswith('png') or filename.endswith('jpeg'):
                        img = Image.open(filename)
                        st.image(img)
                        with open(filename, "rb") as img_file:
                            btn_c1, btn_c2 = st.columns([1, 5])
                            with btn_c1:
                                btn = st.download_button(
                                    label="Download Image",
                                    data=img_file,
                                    file_name=selectedItem,
                                )
                            with btn_c2:
                                delete = st.button("Delete File")
                                if delete:
                                    with st.spinner('Deleting...'):
                                        time.sleep(3)
                                        os.remove(filename)
                                        st.success('Success ✅')
                                        time.sleep(2)

                                        rerun()

                    elif filename.endswith('wav') or filename.endswith('mp3'):
                        st.audio(filename)
                        with open(filename, "rb") as audio_file:
                            btn_c1, btn_c2 = st.columns([1, 5])
                            with btn_c1:
                                btn = st.download_button(
                                    label="Download Audio",
                                    data=audio_file,
                                    file_name=selectedItem

                                )
                            with btn_c2:
                                delete = st.button("Delete File")
                                if delete:
                                    with st.spinner('Deleting...'):
                                        os.remove(filename)
                                        time.sleep(2)
                                        st.success('Succes ✅')
                                        rerun()
                    elif filename.endswith('txt') and 'transcript' in filename:
                        with open('tempDir/transcripts/{}'.format(selectedItem)) as transcript_file:
                            text = st.write(str(transcript_file.read()))
                            btn_c1, btn_c2 = st.columns([1, 4])
                            delete = btn_c2.button("Delete File")
                            f = open(
                                'tempDir/transcripts/{}'.format(selectedItem))
                            btn_c1.download_button(
                                label='Download Transcript', data=f, file_name=selectedItem)
                            if delete:
                                with st.spinner('Deleting...'):
                                    os.remove(
                                        'tempDir/transcripts/{}'.format(selectedItem))
                                    time.sleep(2)
                                    st.success('Succes ✅')
                                    rerun()

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

                            elif filename.endswith('docx'):
                                raw_text = docx2txt.process(text_file)
                                st.write(raw_text)

                            if flag == False:
                                btn = st.download_button(
                                    label="Download Document",
                                    data=text_file,
                                    file_name=selectedItem
                                )
                            delete = st.button("Delete File")
                            if delete:
                                with st.spinner('Deleting...'):
                                    os.remove(filename)
                                    time.sleep(2)
                                    st.success('Succes ✅')
                                    rerun()
        except StreamlitAPIException:
            st.error('You have not uploaded any files yet ❌')

        with st.expander("📈 Raw Output"):
            st.write(filePaths)
            st.write(fileList)
            st.write(extensionList)

    # transcription menu
    if (choice == menu[2]):
        fileList = []
        filePaths = []
        extensionList = []
        flag = False
        languageList = ['en-US', 'en-GB', 'de-DE']

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
        languageList.sort()

        st.header("🌐 Supports up to > 30 Min")

        if not fileList:
            st.error("You have not uploaded any files yet ❌")

        selectedItem = st.selectbox("🔎 Choose File", fileList)
        selectedLanguage = st.selectbox("🔎 Choose Language", languageList)

        matching = [s for s in filePaths if selectedItem in s]
        string = ' '.join(matching)
        realFileName = selectedItem
        head, sep, tail = realFileName.partition('.')
        transcriptFilename = head+'.txt'
        col1, col2 = st.columns([1, 8])
        if col1.button('Transcribe'):
            st.subheader("Transcribed Text:")
            with st.spinner('Transcribing...'):
                time.sleep(2)
                txt = get_large_audio_transcription(
                    string, selectedLanguage, transcriptName=head)
                label_embeddings = np.load("label_embeddings.npy")
                model_name = "distilbert-base-german-cased"
                labels_path = "./labels.csv"
                model_path = "./wandering-sponge-4.pth"
                labels_df = pd.read_csv(labels_path, header=None, index_col=0)
                if "model" not in st.session_state:
                    with torch.no_grad():
                        st.session_state.model = AutoModel.from_pretrained(
                            "distilbert-dlf")

                if "tokenizer" not in st.session_state:
                    st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                        model_name)
                n_keywords = st.slider(
                    "Anzahl der Schlagwörter, die generiert werden sollen.", min_value=1, max_value=15, value=10, step=1)
                input_text = st.text_area(
                    "Text", value=txt, height=500, key=None, help=None, on_change=None, args=None, kwargs=None)
                top_k = get_k_most_similar_keywords(
                    input_text, label_embeddings, st.session_state.model, st.session_state.tokenizer, n_keywords)
                st.dataframe(top_k, height=500)

                # st.caption(txt)
                st.markdown(
                    '<style> .css-12nj2tl small p, .css-12nj2tl small ol, .css-12nj2tl small ul, .css-12nj2tl small dl, .css-12nj2tl small li .css-177yq5e small p, .css-177yq5e small ol, .css-177yq5e small ul, .css-177yq5e small dl, .css-177yq5e small li {font-size: 1.25em; font-weight:400}</style>', unsafe_allow_html=True)
                # Defaults to 'text/plain'
                st.download_button(label='Download Transcript',
                                   data=txt, file_name=transcriptFilename)
                st.success('Done!')

        if col2.button('Live'):
            print('Soon')
            # with sr.Microphone() as source:
            #     # read the audio data from the default microphone
            #     audio_data = r.record(source, duration=5)
            #     st.write("Recognizing...")

            #     # convert speech to text
            #     text = r.recognize_google(audio_data)
            #     st.write(text)

    if (choice == menu[3]):
        st.subheader("About")
        st.info("Dogan Teke, 7335741")
        st.info("Universität zu Köln, 2022")


if __name__ == '__main__':
    main()
