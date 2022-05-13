
from asyncore import write
from tabnanny import check
from pyparsing import col
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
import json
import os.path

import datetime
from random import randrange
from google.cloud import firestore
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
from streamlit.script_request_queue import RerunData
from streamlit.elements import form
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import AutoModel, DistilBertTokenizerFast
from genericpath import exists
from io import BytesIO
from re import M, X
from blinker import base
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from google.oauth2 import service_account

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
    checkSecret = ("cloudKey.json")

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

        if os.path.isfile(checkSecret):
            print('I found :', checkSecret)
        else:
            print('I miss: ', checkSecret)
            msg = st.warning("🚩 JSON Credentials need to be downloaded... ")
            try:
                with st.spinner('Initiating...'):
                    time.sleep(3)
                    url_key = "https://www.dl.dropboxusercontent.com/s/ks3vyqptcsxdl1g/cloudkey.json?dl=0"
                    r_key = requests.get(url_key, allow_redirects=True)
                    open("cloudKey.json", 'wb').write(r_key.content)
                    del r_key
                    msg.success("Download was successful ✅")
            except:
                msg.error("Error downloading model files...😥")

    colT1, colT2 = st.columns([3, 8])
    with colT2:
        st.title("🤖 Transcriptor App")

    # sidebar menu
    menu = ["📂 Upload", "🔎 My Files", "💬 Transcript", "ℹ️ About", "🌟 Rating"]
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
            with st.expander("🗃️ "+selectedItem):
                fileDir = os.path.dirname(os.path.realpath('__file__'))
                filename = os.path.join(fileDir, string)
                agree = st.checkbox('Display Data 📊', key=selectedItem)
                if agree:
                    if filename.endswith('csv'):
                        df = pd.read_csv(filename)
                        st.write(df)
                        csv = convert_df(df)

                        btn_c1, btn_c2 = st.columns([2, 5])

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
                            btn_c1, btn_c2 = st.columns([2, 5])
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
                            btn_c1, btn_c2 = st.columns([2, 5])
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
                            btn_c1, btn_c2 = st.columns([2, 4])
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
        flag = True
        wordList = []

        st.header("🌐 Supports up to > 30 Min")

        if not fileList:
            st.error("You have not uploaded any files yet ❌")

        selectedItem = st.selectbox(
            "🔎 Choose File", (x for x in fileList if '.wav' in x))
        selectedLanguage = st.selectbox(
            "🔎 Choose Language", languageList, key="language")

        matching = [s for s in filePaths if selectedItem in s]
        string = ' '.join(matching)
        realFileName = selectedItem
        head, sep, tail = realFileName.partition('.')
        transcriptFilename = head+'.txt'
        transcriptKeywordsFilename = head+'keywords.txt'

        col1, col2 = st.columns([3, 8])
        selectedWords = []
        if col1.checkbox('📝 Transcribe File'):
            st.subheader("Transcribed Text:")
            with st.spinner('Transcribing...'):
                time.sleep(2)
                txt = get_large_audio_transcription(
                    string, selectedLanguage, transcriptName=head)
                label_embeddings = np.load("label_embeddings.npy")
                model_name = "distilbert-base-german-cased"
                labels_path = "./labels.csv"

                if "model" not in st.session_state:
                    with torch.no_grad():
                        st.session_state.model = AutoModel.from_pretrained(
                            "distilbert-dlf")

                if "tokenizer" not in st.session_state:
                    st.session_state.tokenizer = DistilBertTokenizerFast.from_pretrained(
                        model_name)

            n_keywords = st.slider(
                "Anzahl der Schlagwörter, die generiert werden sollen.", min_value=1, max_value=15, value=10, step=1)
            input_text = st.text_area(
                "Text", value=txt, height=500, key=None, help=None, on_change=None, args=None, kwargs=None)
            top_k = get_k_most_similar_keywords(
                input_text, label_embeddings, st.session_state.model, st.session_state.tokenizer, n_keywords)

            column_1, column_2 = st.columns([3, 3])

            mainFrame = column_1.dataframe(top_k, height=500)

            if wordList != None:
                wordList = []

            for key in top_k['name']:
                wordList.append(key)

            if "keys" not in st.session_state:
                st.session_state['keys'] = wordList
            if 'keys' in st.session_state:
                st.session_state['keys'] = wordList

            with column_2:
                selectedSessionItem = st.empty()
                empty = st.empty()

            empty.multiselect(
                "", st.session_state["keys"], key='sel_key')

            if 'sel_key' in st.session_state:
                selectedSessionItem.write(
                    f'Chosen Words: {st.session_state["sel_key"]}')
                for key in st.session_state['sel_key']:
                    if key not in selectedWords:
                        selectedWords.append(key)
                print('SELECTED: ', selectedWords)

            if 'sel_key1' in st.session_state:
                selectedSessionItem.empty()
                selectedSessionItem.write(
                    f'Chosen Words: {st.session_state["sel_key1"]}')
                for key in st.session_state['sel_key1']:
                    if key not in selectedWords:
                        selectedWords.append(key)
                print('SELECTED: ', selectedWords)

            confirm = column_2.checkbox("⏏️ Reload Keys")

            if confirm:
                with st.spinner('Reload model...'):
                    empty.empty()
                    del st.session_state["keys"]
                    if 'keys' not in st.session_state:
                        st.session_state['keys'] = wordList
                    if 'keys' in st.session_state:
                        st.session_state['keys'] = wordList
                    column_1.subheader('New Results 📉')
                    btn_col1, btn_col2 = st.columns([4, 11])
                    time.sleep(2)
                    retrained_top_k = retrainModel(selectedWords,
                                                   input_text, label_embeddings, st.session_state.model, st.session_state.tokenizer, n_keywords)

                    for key in retrained_top_k['name']:
                        wordList.append(key)
                        st.session_state['keys'].append(key)

                    st.session_state['keys'] = list(
                        dict.fromkeys(st.session_state['keys']))
                    st.session_state['keys'].sort()
                    wordList = list(dict.fromkeys(wordList))
                    selectedWords = list(dict.fromkeys(selectedWords))
                    wordList = list(set(wordList))
                    selectedWords = list(set(selectedWords))
                    wordList.sort()
                    selectedWords.sort()
                    empty.multiselect(
                        "", st.session_state["keys"], key='sel_key1')

                    column_1.dataframe(
                        retrained_top_k, height=500)

                    btn_col1.download_button(label='Download Transcript',
                                             data=txt, file_name=transcriptFilename)
                    btn_col2.download_button(label='Download Keywords',
                                             data=convert_df(retrained_top_k), file_name=transcriptKeywordsFilename)

                    st.markdown(
                        '<style> .css-12nj2tl small p, .css-12nj2tl small ol, .css-12nj2tl small ul, .css-12nj2tl small dl, .css-12nj2tl small li .css-177yq5e small p, .css-177yq5e small ol, .css-177yq5e small ul, .css-177yq5e small dl, .css-177yq5e small li {font-size: 1.25em; font-weight:400} </style>', unsafe_allow_html=True)
                    successmsg = column_2.success('🔑 Loaded new keys')

                    columnOne, columnTwo = st.columns([2, 2])
                    columnOne.subheader('🗑️ Removed Keys')

                    for x in selectedWords:
                        columnOne.caption(x)
                if successmsg is not None:
                    print()
                else:
                    columnOne.success('Done')

        if col2.checkbox('🔴 Transcribe Live'):
            print('LAN: ', selectedLanguage)
            stt_button = Button(label="Speak", width=100)
            stt_button.js_on_event("button_click", CustomJS(code="""
					var recognition = new webkitSpeechRecognition();
					recognition.continuous = true;
					recognition.interimResults = true;
					recognition.lang = "de-DE"


					recognition.onresult = function (e) {
						var value = "";
						for (var i = e.resultIndex; i < e.results.length; ++i) {
							if (e.results[i].isFinal) {
								value += e.results[i][0].transcript;
							}
						}
						if ( value != "") {
							document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
						}
					}
					recognition.start();
					"""))

            result = streamlit_bokeh_events(
                stt_button,
                events="GET_TEXT",
                key="listen",
                refresh_on_update=False,
                override_height=75,
                debounce_time=0)

            if result:
                if "GET_TEXT" in result:
                    text = result.get("GET_TEXT")
                    model_name = "distilbert-base-german-cased"
                    label_embeddings = np.load("label_embeddings.npy")

                    if "model" not in st.session_state:
                        with torch.no_grad():
                            st.session_state.model = AutoModel.from_pretrained(
                                "distilbert-dlf")

                    if "tokenizer" not in st.session_state:
                        st.session_state.tokenizer = DistilBertTokenizerFast.from_pretrained(
                            model_name)

                    n_keywords = st.slider(
                        "Anzahl der Schlagwörter, die generiert werden sollen.", min_value=1, max_value=15, value=10, step=1)
                    input_text = st.text_area(
                        "Text", value=text, height=500, key=None, help=None, on_change=None, args=None, kwargs=None)
                    top_k = get_k_most_similar_keywords(
                        input_text, label_embeddings, st.session_state.model, st.session_state.tokenizer, n_keywords)
                    st.dataframe(top_k)

    if (choice == menu[3]):
        st.subheader("About")
        st.info("Dogan Teke, 7335741")
        st.info("Universität zu Köln, 2022")
        st.info("https://github.com/lafllamme/transcriptapp")

    if (choice == menu[4]):

        col1, col2, col3 = st.columns([2, 2, 1])
        col2.subheader("Rate this app 🤩")
        efficiency = st.slider('🛠️ Efficiency', 1, 5)
        velocity = st.slider('🏎️ Velocity', 1, 5)
        accuracy = st.slider('🎯 Accuracy', 1, 5)
        usability = st.slider('📲 Usability', 1, 5)
        design = st.slider('✂️ Design', 1, 5)
        overallRating = (efficiency + velocity +
                         accuracy + usability + design) / 5

        col1, col2, col3 = st.columns([2, 0.2, 2])
        ratingArr = []
        score = 0

        with col1:
            st.empty()
            st.text("")
            st.text("")
            st.markdown("Overall Result:")
        with col2:
            if 0 <= overallRating < 1.5:
                st.text("")
                st.text("")
                st.markdown("[⭐]")
                ratingArr.append(1)
                score = overallRating
            if 1.5 <= overallRating < 2.5:
                st.text("")
                st.text("")
                st.markdown("[⭐⭐]")
                ratingArr.append(2)
                score = overallRating

            if 2.5 <= overallRating < 3.5:
                st.text("")
                st.text("")
                st.markdown("[⭐⭐⭐]")
                ratingArr.append(3)
                score = overallRating

            if 3.5 <= overallRating < 4.5:
                st.text("")
                st.text("")
                st.markdown("[⭐⭐⭐⭐]")
                ratingArr.append(4)
                score = overallRating

            if 4.5 <= overallRating <= 5:
                st.text("")
                st.text("")
                st.markdown("[⭐⭐⭐⭐⭐]")
                ratingArr.append(5)
                score = overallRating

        with col3:
            st.empty()

        st.subheader("Your Statics:")
        st.write("You chose {} ⭐".format(ratingArr[0]))
        st.write("Float score", score)
        st.write("Sliderdata: ", [efficiency, velocity,
                                  accuracy,
                                  usability,
                                  design])

        name = st.text_input('Your name', '...')
        comment = st.text_area("Your feedback", "...")
        submit = st.button("Send Rating")
        documentId = name+"#ID:"+str(randrange(100))
        notAvailable = False

        # Authenticate to Firestore with the JSON account key.
        if os.path.isfile('.streamlit/secrets.toml'):
            print(" I found secrets :)")
            # Do something with the file
        else:
            print("No Secrets :'(")
            notAvailable = True

        if(notAvailable == True):
            db = firestore.Client.from_service_account_json("cloudKey.json")
            print('Using downloaded JSON Config')
        else:
            print('Using streamlit secrets Config')
            key_dict = json.loads(st.secrets["textkey"])
            creds = service_account.Credentials.from_service_account_info(
                key_dict)
            db = firestore.Client(
                credentials=creds, project="transcript-app-338213")

        if name and comment and submit:
            now = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")  # current date and time
            with st.spinner('Wait for it...'):
                time.sleep(2)
                doc_ref = db.collection("ratings").document(documentId)
                doc_ref.set({
                    "name": name,
                    "comment": comment,
                    "stars": ratingArr[0],
                    "points": overallRating,
                    "efficiency": efficiency,
                    "velocity": velocity,
                    "accuracy": accuracy,
                    "usability": usability,
                    "design": design,
                    "date": now
                })
            st.success('Done!')

        col1, col2, col3 = st.columns([2, 2, 1])
        col2.subheader("Latest Reviews")
        rating_ref = db.collection("ratings")

        currentAverage = []
        for rating in rating_ref.stream():
            userRating = rating.to_dict()
            name = userRating["name"]
            stars = userRating["stars"]
            average = userRating["points"]
            date = userRating["date"]
            comment = userRating["comment"]
            acc = userRating["accuracy"]
            eff = userRating["efficiency"]
            usa = userRating["usability"]
            velo = userRating["velocity"]
            des = userRating["design"]
            currentAverage.append(average)
            title = name + ", " + date

            with st.expander(title):
                st.markdown(f"**Reviewer**: _{name}_")
                st.markdown(f"**Reviewer's Point Average**: _{average}_")
                st.markdown(f"**Comment**: _{comment}_")
                st.markdown(f"**Result was**: _{stars} ⭐_")
                detailed = st.checkbox("Detailed", key=name)
                if detailed:
                    st.markdown("**🎯 Accuracy** _{}_".format(acc))
                    st.markdown("**🛠️ Efficiency** _{}_".format(eff))
                    st.markdown("**📲 Usability** _{}_".format(usa))
                    st.markdown("**🏎️ Velocity** _{}_".format(velo))
                    st.markdown("**✂️ Design** _{}_".format(des))

        displayAvg = sum(currentAverage) / len(currentAverage)
        col1, col2, col3 = st.columns([2, 2, 1])
        col2.markdown(f"_Current Average is {displayAvg} ⭐_")


if __name__ == '__main__':
    main()
