FROM ubuntu:latest

# Set timezone so package doesn't ask
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#install git and py dependencies
RUN apt-get update && apt-get -y install git \
    software-properties-common \
    ffmpeg \
    build-essential \
    libasound-dev \ 
    portaudio19-dev \
    python3-pyaudio 

RUN add-apt-repository ppa:deadsnakes/ppa
# Update aptitude with new repo
RUN apt-get update && apt-get -y install python3.9 \
    python3-pip

#clone latest project
RUN git clone https://github.com/lafllamme/transcriptapp

#go to dir
WORKDIR /transcriptapp
RUN pip3 install -r requirements.txt
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]