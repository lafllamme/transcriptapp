
from asyncore import write
import speech_recognition as sr
import streamlit as st
import pandas as pd
import numpy as np
import shutil
import pdfplumber
import os
import datetime
import pyttsx3

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
from streamlit.elements import form
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import AutoModel, TFAutoModel, AutoTokenizer
from genericpath import exists
from io import BytesIO
from re import M, X
from blinker import base
import warnings
# initialize the speech recognizer
r = sr.Recognizer()
#disabled parallel tokens
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)


def rerun():
    raise RerunException(st.script_request_queue.RerunData(None))


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

                # converts an uploaded .mp3 to .wav for sr support
                elif uploaded_file.name.endswith('mp3'):
                    audio = AudioSegment.from_mp3(uploaded_file)
                    head, sep, tail = uploaded_file.name.partition('.')
                    uploaded_file.name = head + '.wav'
                    file_type = 'wav'

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

# a function that splits the audio file into chunks
# and applies speech recognition

@st.cache
def get_large_audio_transcription(path, language, transcriptName):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """

    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
                              # experiment with this value for your target audio file
                              min_silence_len=500,
                              # adjust this per requirement
                              silence_thresh=sound.dBFS-14,
                              # keep the silence for 1 second, adjustable as well
                              keep_silence=500,
                              )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened, language=language)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text

    text_file = open(
        "tempDir/transcripts/transcript_{}_{}.txt".format(transcriptName, datetime.datetime.now()), "w+")
    # write string to file
    n = text_file.write(whole_text)
    # close file
    text_file.close()
    # return the text for all chunks detected
    shutil.rmtree(folder_name)
    return whole_text


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
