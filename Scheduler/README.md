# PeopleWelcome Image Classifier

An scheduler module for PeopleWelcome

To run on Ubuntu 22.04:

## prerequisites:
sudo apt update

sudo apt install python3-pip

pip install --upgrade pip

pip install tensorflow

if the following two commands don't work, try them with sudo 

pip install fastapi 

pip install "uvicorn[standard]"

## How to start the server
we are using a non-default port for this API

uvicorn main:app --reload --port 3003

You can call APIs in the web browser
http://127.0.0.1:3003/docs
