  
# **Project Libraries:**

 - *[Streamlit Framework](https://streamlit.io/)* 
 - *[Python 3](https://www.python.org/)*
 - [*Virtualenv*](https://docs.python.org/3/tutorial/venv.html)
 - *[Docker](https://www.docker.com/products/docker-desktop)*
 - *[Firebase](https://firebase.google.com/)*



# Info

*This is a transcriptor app written in Python 3 with streamlit.
It supports transcription of audio files in english or german, including a keyword generator for your content. You just need to upload .mp3 or .wav files in the upload section.*

# **Setup**

## **Locally**

 1. Install Python (*preferably v3.x +*) 
 2. Install Pip3 (*python package manager*)
 3. [Install Virtualenv](https://docs.python.org/3/tutorial/venv.html)
 4. Create python virtualenv inside root dir
 > `python3 -m venv env`

 6.  Activate it

> `source env/bin/activate`

 7. Install the packages from packages.txt with your cmd

>  `sudo apt-get install ...`

 8. Install the requirements inside requirements.txt:

>  `pip3 install -r requirements.txt`

9. Run the following command to start the app:
> `streamlit run app.py`

## Docker

 1. Install latest [Docker Version](https://www.docker.com/products/docker-desktop) for your pc
 2.  Apply this settings *(this is **NOT** default*, *ignore img size*)
	 ![setup](https://i.imgur.com/DXgntoB.png =450x250)

 3. Build the container:

 

> `docker build -t streamlitapp:latest .`
4. Run the container with default port :
> `docker run -p 8501:8501 streamlitapp:latest`

Now you should enter the app on http://localhost:8501/

## Online

I have uploaded the latest version of my app on the streamlit cloud under https://share.streamlit.io/lafllamme/transcriptapp/main/app.py.

**[Click here to checkout the current demo!](https://share.streamlit.io/lafllamme/transcriptapp/main/app.py)**
Rating and / or feedback is appreciated :)




# Note

 - If you install the app for the first time, it needs to download the
   model files first in order to work, you'll see an appropriate
   notification
 - If you build the container for the first time, it could take a while  (*approx. 2mins*)

	![model download](https://i.imgur.com/Sbd57Ly.png =450x125)

> `{ "type": "service_account", "project_id": "transcript-app-338213", "private_key_id": "xxxxx", "private_key": "-----BEGIN PRIVATE KEY-----\xxxx-----END PRIVATE KEY-----\n", "client_email": "firebase-xxxx@xxx-app-338213.iam.gserviceaccount.com", "client_id": "xxx", "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-yaesk%40transcript-app-338213.iam.gserviceaccount.com" }`
 - ~~If you build locally or through docker, you need to add a [.json configuration](https://www.dropbox.com/s/ks3vyqptcsxdl1g/cloudkey.json?dl=0) under root/**cloudKey.json** for a firebase connection~~
 ***- This should download now automatically, if no secrets set***
									![dir](https://i.imgur.com/kADBlaD.png =250x350)
