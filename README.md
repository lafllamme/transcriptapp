# **Project Libraries:**

 - *[Streamlit Framework](https://streamlit.io/)* https://github.com/lafllamme/transcriptapp
 - *[Python 3](https://www.python.org/)*
 - *[Docker](https://www.docker.com/products/docker-desktop)*


# **Setup**

## **Locally**

 1. Install Python (*preferably v3.x +*) 
 2. Install Pip3 (*python package manager*)
 3. Install the packages from packages.txt with your cmd

>  `sudo apt-get install ...`

 4. Install the requirements inside requirements.txt:

>  `pip3 install -r requirements.txt`

5. Run the following command to start the app:
> `streamlit run app.py`

## Docker

 1. Install latest [Docker Version](https://www.docker.com/products/docker-desktop) for your pc
 2.  Apply this settings *(this is **NOT** default*, *ignore img size*)
 ![setup](https://i.imgur.com/DXgntoB.png)

 3. Build the container:

 

> `docker build -t streamlitapp:latest .`
4. Run the container with default port :
> `docker run -p 8501:8501 streamlitapp:latest`

Now you should enter the app on http://localhost:8501/

## Online

I have uploaded the latest version of my app on the streamlit cloud under https://share.streamlit.io/lafllamme/transcriptapp/main/app.py.

**[Click here to checkout the current demo!](https://share.streamlit.io/lafllamme/transcriptapp/main/app.py)**
Rating and / or feedback is appreciated :)


