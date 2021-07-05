# clickbait_detection
This project is for my summer internship at Somera and I will be creating a clickbait detector using BERT with the checkpoint [Turkish Bert Model improved by MDZ Digital Library Team](https://huggingface.co/dbmdz/bert-base-turkish-cased). Simple UI with REST API is created and dockerization is completed. 

**To create a docker container and then run it:**

After cloning this repo, you need to build docker by writing this command in the same path as your project directory; docker build -t _clickbait_ . 

This command uses the Dockerfile to build a new container image. Afterwards, you can open the docker desktop (you need to install Docker Desktop), images and click the run button near _clickbait_ image. 

**To run the app in your local computer:**

After cloning this repo, you can simply run app.py file and it will automatically load the model and will give a link so that user can try the web app. You can just click the link and try. But you need some libraries to run the python file and these libraries with their version names are written on requirements.txt file.
